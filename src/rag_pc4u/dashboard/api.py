"""
Dashboard FastAPI — Nextcloud Sync Manager.

Fournit :
  - La page HTML du dashboard (GET /)
  - Une API REST pour gérer les mappings et déclencher des syncs
  - Un endpoint SSE pour les mises à jour en temps réel (toutes les 2s)

Changements v2 :
  - remote_path (str) → remote_paths (list[str]) — point 2
  - Vérification unicité label/collection — point 1
  - GET /api/mappings/search — recherche par label ou collection — point 1
  - interval_minutes supporte jusqu'à 10080 (1 semaine) — point 6
  - DELETE /api/mappings/{id} nettoie le cache Nextcloud local — point 8
  - MappingCreate/Update migrés vers remote_paths
"""

import asyncio
import json
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from rag_pc4u.core.config import settings
from rag_pc4u.core.tz_utils import now_paris_naive
from rag_pc4u.dashboard import state as ds
from rag_pc4u.dashboard.scheduler import SyncScheduler
from rag_pc4u.ingestion.nextcloud_watcher import NextcloudWatcher

logger = structlog.get_logger(__name__)

# Singletons

_scheduler: Optional[SyncScheduler] = None
_watcher: Optional[NextcloudWatcher] = None
_in_progress: dict[str, bool] = {}


# Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler, _watcher
    _watcher = NextcloudWatcher()
    _scheduler = SyncScheduler()

    for mid, mapping in ds.get_mappings().items():
        if mapping.get("active"):
            _register_job(mid, mapping, run_immediately=False)
            logger.info("dashboard.job_restored", mapping_id=mid)

    logger.info("dashboard.started", active_jobs=len(_scheduler.list_jobs()))
    yield
    _scheduler.shutdown()
    logger.info("dashboard.stopped")


app = FastAPI(
    title="RAG PC4U — Nextcloud Dashboard",
    description="Gestion des syncs Nextcloud → Collections RAG",
    version="2.0.0",
    lifespan=lifespan,
)

CURRENT_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=CURRENT_DIR / "static"), name="static")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _register_job(
    mapping_id: str,
    mapping: dict,
    run_immediately: bool = True,
) -> None:
    """Enregistre un job de sync APScheduler pour un mapping."""

    def _job():
        _execute_sync(mapping_id)

    start_date = None
    raw_start_at = mapping.get("start_at")
    if raw_start_at:
        try:
            start_date = datetime.fromisoformat(raw_start_at)
        except ValueError:
            logger.warning("dashboard.invalid_start_at_format", start_at=raw_start_at)

    _scheduler.add_job(
        job_id=mapping_id,
        sync_fn=_job,
        interval_minutes=mapping["interval_minutes"],
        run_immediately=run_immediately,
        start_date=start_date,
    )


def _execute_sync(mapping_id: str) -> dict:
    """
    Exécute une sync et persiste les résultats.
    Appelé soit par le scheduler (thread), soit par BackgroundTasks (thread pool).
    """
    mapping = ds.get_mapping(mapping_id)
    if not mapping:
        return {"status": "error", "error_message": "Mapping introuvable"}

    _in_progress[mapping_id] = True
    try:
        # Point 2 : on passe remote_paths (list) au watcher.
        # Rétrocompatibilité : _normalize_mapping dans state.py garantit
        # que même les anciens mappings avec "remote_path" str ont bien
        # un champ "remote_paths" list ici.
        stats = _watcher.sync(
            remote_paths=mapping["remote_paths"],
            collection_name=mapping["collection_name"],
        )
    except Exception as e:
        logger.exception("dashboard.sync_unhandled_error", mapping_id=mapping_id)
        stats = {
            "status": "error",
            "error_message": str(e),
            "finished_at": now_paris_naive().isoformat(),
            "new": 0,
            "modified": 0,
            "deleted": 0,
            "errors": 1,
        }
    finally:
        _in_progress[mapping_id] = False

    ds.update_mapping_after_sync(mapping_id, stats)
    ds.add_sync_record(mapping_id, stats)
    return stats


def _cleanup_mapping_cache(mapping: dict) -> None:
    """
    Point 8 — Nettoyage du cache Nextcloud local lors de la suppression
    d'un mapping. Supprime le dossier de cache et les fichiers d'état
    WebDAV (.nextcloud_etag_*.json) associés à ce mapping.
    """
    collection_name = mapping.get("collection_name")
    remote_paths = mapping.get("remote_paths", [])

    # 1. Cache local des fichiers téléchargés
    if collection_name:
        cache_dir = Path(__file__).parent / "nextcloud_cache" / collection_name
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                logger.info(
                    "dashboard.cache_cleaned",
                    collection=collection_name,
                    path=str(cache_dir),
                )
            except Exception as e:
                logger.warning(
                    "dashboard.cache_cleanup_failed",
                    collection=collection_name,
                    error=str(e),
                )

    # 2. Fichiers d'état ETags WebDAV par chemin distant
    state_base = Path(__file__).parent / "fichier_injecter"
    for remote_path in remote_paths:
        safe = remote_path.replace("/", "_").replace(":", "_").replace(" ", "_").strip("_")
        etag_file = state_base / f".nextcloud_etag_{safe}.json"
        if etag_file.exists():
            try:
                etag_file.unlink()
                logger.info("dashboard.etag_cleaned", path=str(etag_file))
            except Exception as e:
                logger.warning("dashboard.etag_cleanup_failed", path=str(etag_file), error=str(e))

    # 3. Fichier d'état d'ingestion run.py
    if collection_name:
        safe_coll = collection_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        ingestion_state = (
            Path(__file__).parent.parent / "ingestion" /
            f"fichier_injecter/.ingestion_state_{safe_coll}.json"
        )
        if ingestion_state.exists():
            try:
                ingestion_state.unlink()
                logger.info("dashboard.ingestion_state_cleaned", path=str(ingestion_state))
            except Exception as e:
                logger.warning(
                    "dashboard.ingestion_state_cleanup_failed",
                    path=str(ingestion_state),
                    error=str(e),
                )


# ── Schémas Pydantic ──────────────────────────────────────────────────────────

class MappingCreate(BaseModel):
    remote_paths: List[str] = Field(
        ...,
        min_length=1,
        description="Liste de chemins Nextcloud (dossiers ou fichiers individuels).",
    )
    collection_name: str = Field(..., description="Nom de la collection Qdrant cible")
    # Point 6 : limite haute portée à 10080 min = 7 jours (hebdomadaire)
    interval_minutes: int = Field(default=15, ge=1, le=10080)
    label: Optional[str] = Field(default=None, description="Libellé affiché dans le dashboard")
    start_at: Optional[str] = Field(default=None, description="Date/heure ISO de la 1ère sync planifiée")


class MappingUpdate(BaseModel):
    remote_paths: Optional[List[str]] = Field(default=None, description="Chemins Nextcloud")
    collection_name: Optional[str] = Field(default=None)
    interval_minutes: Optional[int] = Field(default=None, ge=1, le=10080)
    label: Optional[str] = Field(default=None)
    start_at: Optional[str] = Field(default=None)


class SyncResponse(BaseModel):
    status: str
    mapping_id: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# Status

@app.get("/api/status", tags=["Infos"])
async def api_status():
    connected = _watcher.test_connection()
    counts = ds.get_sync_counts_today()
    return {
        "status": "ok",
        "nextcloud_connected": connected,
        "nextcloud_url": settings.nextcloud_url,
        "nextcloud_user": settings.nextcloud_user,
        "active_jobs": len(_scheduler.list_jobs()),
        "syncs_today": counts["total"],
        "errors_today": counts["errors"],
        "timestamp": now_paris_naive().isoformat(),
    }


# Mappings

@app.get("/api/mappings", tags=["Mappings"])
async def api_list_mappings():
    """Liste tous les mappings avec leur statut live (in_progress, next_run)."""
    mappings = ds.get_mappings()
    jobs = {j["id"]: j for j in _scheduler.list_jobs()}
    return [
        {
            **m,
            "in_progress": _in_progress.get(mid, False),
            "next_run": jobs.get(mid, {}).get("next_run"),
        }
        for mid, m in mappings.items()
    ]


@app.get("/api/mappings/search", tags=["Mappings"])
async def api_search_mappings(
    label: Optional[str] = Query(default=None, description="Sous-chaîne du libellé (insensible à la casse)"),
    collection: Optional[str] = Query(default=None, description="Nom exact de la collection Qdrant"),
):
    """
    Point 1 — Recherche de mappings par libellé (sous-chaîne) et/ou
    par nom de collection (exact). Les deux filtres sont cumulatifs.
    """
    return ds.search_mappings(label=label, collection_name=collection)


@app.post("/api/mappings", tags=["Mappings"], status_code=201)
async def api_create_mapping(body: MappingCreate, bg: BackgroundTasks):
    """
    Crée un nouveau mapping.
    Point 1 : rejette les doublons de label ou de collection.
    Point 2 : accepte plusieurs chemins (remote_paths).
    Point 6 : intervalle jusqu'à 7 jours (10080 min).
    """
    result = ds.add_mapping(
        remote_paths=body.remote_paths,
        collection_name=body.collection_name,
        interval_minutes=body.interval_minutes,
        label=body.label,
        start_at=body.start_at,
    )
    if isinstance(result, str):
        raise HTTPException(409, result)

    _register_job(result["id"], result, run_immediately=False)
    if not body.start_at:
        bg.add_task(_execute_sync, result["id"])
    return result


@app.put("/api/mappings/{mapping_id}", tags=["Mappings"])
async def api_update_mapping(mapping_id: str, body: MappingUpdate):
    """
    Modifie un mapping. Bloque si une sync est en cours.
    Reconstruit le job APScheduler pour que tout changement d'intervalle
    ou de start_at soit immédiatement effectif.
    """
    if _in_progress.get(mapping_id):
        raise HTTPException(409, "Une sync est en cours — attendez qu'elle se termine")

    if not ds.get_mapping(mapping_id):
        raise HTTPException(404, "Mapping introuvable")

    result = ds.update_mapping(
        mapping_id,
        remote_paths=body.remote_paths,
        collection_name=body.collection_name,
        interval_minutes=body.interval_minutes,
        label=body.label,
        start_at=body.start_at,
    )
    if result is None:
        raise HTTPException(404, "Mapping introuvable")
    if isinstance(result, str):
        raise HTTPException(409, result)

    _scheduler.remove_job(mapping_id)
    _register_job(mapping_id, result, run_immediately=False)
    return result


@app.delete("/api/mappings/{mapping_id}", tags=["Mappings"])
async def api_delete_mapping(mapping_id: str):
    """
    Supprime un mapping, arrête son job scheduler, et nettoie le cache
    Nextcloud local associé (point 8).
    """
    if _in_progress.get(mapping_id):
        raise HTTPException(409, "Une sync est en cours — attendez qu'elle se termine")

    mapping = ds.get_mapping(mapping_id)
    if not mapping:
        raise HTTPException(404, "Mapping introuvable")

    _scheduler.remove_job(mapping_id)

    # Point 8 : nettoyage cache avant suppression de l'état
    _cleanup_mapping_cache(mapping)

    if not ds.delete_mapping(mapping_id):
        raise HTTPException(404, "Mapping introuvable")

    _in_progress.pop(mapping_id, None)
    return {"deleted": mapping_id}


# Sync manuelle

@app.post("/api/sync/{mapping_id}", tags=["Sync"])
async def api_trigger_sync(mapping_id: str, bg: BackgroundTasks):
    """Déclenche une sync immédiate en arrière-plan."""
    if not ds.get_mapping(mapping_id):
        raise HTTPException(404, "Mapping introuvable")
    if _in_progress.get(mapping_id):
        return SyncResponse(status="already_running", mapping_id=mapping_id)
    bg.add_task(_execute_sync, mapping_id)
    return SyncResponse(status="started", mapping_id=mapping_id)


# Historique

@app.get("/api/history", tags=["Historique"])
async def api_history(
    limit: int = 50,
    mapping_id: Optional[str] = None,
):
    return ds.get_sync_history(limit=limit, mapping_id=mapping_id)


# Navigateur Nextcloud

@app.get("/api/nextcloud/browse", tags=["Nextcloud"])
async def api_browse(path: str = "/"):
    """Explore l'arborescence Nextcloud (dossiers + fichiers)."""
    try:
        dirs = _watcher.list_remote_dirs(path)
        files_raw = _watcher.list_remote_files(path)
        files = [{"name": f["name"], "size": f["size"]} for f in files_raw]

        from pathlib import PurePosixPath
        parent = str(PurePosixPath(path).parent)
        parent = None if parent == path or (parent == "/" and path == "/") else parent

        return {
            "path": path,
            "parent": parent,
            "directories": dirs,
            "files": files,
            "file_count": len(files),
        }
    except Exception as e:
        raise HTTPException(500, f"Erreur Nextcloud : {e}")


# SSE mises à jour temps réel

@app.get("/api/events", tags=["Live"], include_in_schema=False)
async def api_sse():
    async def generator():
        jobs_cache: dict[str, str] = {}
        iteration = 0

        while True:
            if iteration % 10 == 0:
                jobs_cache = {j["id"]: j for j in _scheduler.list_jobs()}
            iteration += 1

            mappings = ds.get_mappings()
            payload = [
                {
                    "id": mid,
                    "in_progress": _in_progress.get(mid, False),
                    "last_sync": m.get("last_sync"),
                    "last_status": m.get("last_status"),
                    "last_stats": m.get("last_stats"),
                    "next_run": jobs_cache.get(mid, {}).get("next_run"),
                }
                for mid, m in mappings.items()
            ]

            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
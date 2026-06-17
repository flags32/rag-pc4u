"""
Dashboard FastAPI — Nextcloud Sync Manager.

Fournit :
  - La page HTML du dashboard (GET /)
  - Une API REST pour gérer les mappings et déclencher des syncs
  - Un endpoint SSE pour les mises à jour en temps réel (toutes les 2s)

Architecture :
  - Un NextcloudWatcher partagé (session HTTP persistante)
  - Un SyncScheduler APScheduler (jobs en arrière-plan)
  - Un dict _in_progress pour le statut "live" des syncs
  - Le state.py pour la persistance (mappings + historique)
"""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException
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
# mapping_id → True si une sync est en cours (GIL-safe en CPython)
_in_progress: dict[str, bool] = {}


# Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler, _watcher
    _watcher = NextcloudWatcher()
    _scheduler = SyncScheduler()

    # Restaure les jobs depuis l'état persisté, sans exécution immédiate
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
    version="1.0.0",
    lifespan=lifespan,
)

CURRENT_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=CURRENT_DIR / "static"), name="static")


# Helpers

def _register_job(
    mapping_id: str,
    mapping: dict,
    run_immediately: bool = True,
) -> None:
    """Enregistre un job de sync APScheduler pour un mapping."""

    def _job():
        _execute_sync(mapping_id)

    _scheduler.add_job(
        job_id=mapping_id,
        sync_fn=_job,
        interval_minutes=mapping["interval_minutes"],
        run_immediately=run_immediately,
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
        stats = _watcher.sync(
            remote_path=mapping["remote_path"],
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


# Schémas Pydantic

class MappingCreate(BaseModel):
    remote_path: str = Field(..., description="Chemin relatif sur Nextcloud, ex: /documents/technique")
    collection_name: str = Field(..., description="Nom de la collection Qdrant cible")
    interval_minutes: int = Field(default=15, ge=1, le=1440)
    label: Optional[str] = Field(default=None, description="Libellé affiché dans le dashboard")


class SyncResponse(BaseModel):
    status: str
    mapping_id: Optional[str] = None


# Endpoints

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    """Sert le dashboard HTML."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# Status

@app.get("/api/status", tags=["Infos"])
async def api_status():
    """
    Retourne le statut général du dashboard :
    - Connexion Nextcloud (test en direct)
    - Nombre de jobs actifs
    - Statistiques du jour
    """
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


@app.post("/api/mappings", tags=["Mappings"], status_code=201)
async def api_create_mapping(body: MappingCreate, bg: BackgroundTasks):
    """
    Crée un nouveau mapping, enregistre le job et démarre la 1ère sync
    immédiatement en arrière-plan.
    """
    mapping = ds.add_mapping(
        remote_path=body.remote_path,
        collection_name=body.collection_name,
        interval_minutes=body.interval_minutes,
        label=body.label,
    )
    _register_job(mapping["id"], mapping, run_immediately=False)
    # 1ère sync immédiate en arrière-plan pour ne pas bloquer la réponse
    bg.add_task(_execute_sync, mapping["id"])
    return mapping


@app.delete("/api/mappings/{mapping_id}", tags=["Mappings"])
async def api_delete_mapping(mapping_id: str):
    """Supprime un mapping et arrête son job scheduler."""
    if _in_progress.get(mapping_id):
        raise HTTPException(409, "Une sync est en cours — attendez qu'elle se termine")
    _scheduler.remove_job(mapping_id)
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
    """Retourne l'historique des syncs, filtrable par mapping."""
    return ds.get_sync_history(limit=limit, mapping_id=mapping_id)


# Navigateur Nextcloud

@app.get("/api/nextcloud/browse", tags=["Nextcloud"])
async def api_browse(path: str = "/"):
    """
    Explore l'arborescence Nextcloud.
    Utilisé par le file-picker du dashboard pour choisir le dossier source.
    """
    try:
        dirs = _watcher.list_remote_dirs(path)
        files_raw = _watcher.list_remote_files(path)
        files = [{"name": f["name"], "size": f["size"]} for f in files_raw]

        # Chemin parent (pour le bouton "Retour")
        from pathlib import PurePosixPath
        parent = str(PurePosixPath(path).parent)
        parent = None if parent == path or parent == "/" and path == "/" else parent

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
    """
    Server-Sent Events : pousse les mises à jour de statut toutes les 2s.
    Le client JS reconnecte automatiquement si la connexion est perdue.

    Payload : liste de { id, in_progress, last_sync, last_status,
                          last_stats, next_run }
    """

    async def generator():
        jobs_cache: dict[str, str] = {}  # updated chaque 10 itérations pour éviter l'overhead
        iteration = 0

        while True:
            # Mise à jour du cache des jobs toutes les 10 itérations (~20s)
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
            "X-Accel-Buffering": "no",   # désactive le buffer Nginx si présent
        },
    )
"""
Dashboard FastAPI — Nextcloud Sync Manager.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional
import pytz

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from rag_pc4u.core.config import settings
from rag_pc4u.dashboard import state as ds
from rag_pc4u.dashboard.scheduler import SyncScheduler
from rag_pc4u.ingestion.nextcloud_watcher import NextcloudWatcher

logger = structlog.get_logger(__name__)

_scheduler: Optional[SyncScheduler] = None
_watcher: Optional[NextcloudWatcher] = None
_in_progress: dict[str, bool] = {}

PARIS_TZ = pytz.timezone("Europe/Paris")


class MappingCreate(BaseModel):
    remote_path: str = Field(..., description="Chemin relatif sur Nextcloud")
    collection_name: str = Field(..., description="Nom de la collection Qdrant cible")
    interval_minutes: int = Field(default=15, ge=1, le=1440)
    label: Optional[str] = Field(default=None, description="Libellé optionnel")
    start_at: Optional[str] = Field(default=None, description="Date/Heure au format ISO local YYYY-MM-DDTHH:MM")


def _execute_sync(mapping_id: str) -> None:
    """Wrapper synchrone d'exécution de la sync."""
    if _in_progress.get(mapping_id):
        logger.warning("dashboard.sync_already_running", mapping_id=mapping_id)
        return

    _in_progress[mapping_id] = True
    logger.info("dashboard.sync_started", mapping_id=mapping_id)

    start_time = datetime.now()
    stats = {"status": "success", "new": 0, "modified": 0, "deleted": 0, "errors": 0}

    try:
        mappings = ds.get_mappings()
        if mapping_id not in mappings:
            raise ValueError(f"Mapping {mapping_id} introuvable")

        m = mappings[mapping_id]

        # Simulation d'appel à votre watcher
        res = _watcher.sync(
            remote_path=m["remote_path"],
            collection_name=m["collection_name"]
        )

        stats.update({
            "new": res.get("new", 0),
            "modified": res.get("modified", 0),
            "deleted": res.get("deleted", 0),
            "errors": res.get("errors", 0),
        })
        ds.update_mapping_sync_status(mapping_id, "success", stats)

    except Exception as e:
        logger.exception("dashboard.sync_failed", mapping_id=mapping_id, error=str(e))
        stats.update({"status": "error", "error_message": str(e), "errors": 1})
        ds.update_mapping_sync_status(mapping_id, "error", stats)
    finally:
        _in_progress[mapping_id] = False
        end_time = datetime.now()
        stats.update({
            "started_at": start_time.isoformat(),
            "finished_at": end_time.isoformat()
        })
        ds.add_sync_history_record(mapping_id, stats)
        logger.info("dashboard.sync_finished", mapping_id=mapping_id, stats=stats)


def _register_job(mapping_id: str, mapping: dict, run_immediately: bool = True) -> None:
    """Enregistre un job de sync APScheduler pour un mapping avec gestion fuseau horaire."""
    def _job():
        _execute_sync(mapping_id)

    start_date = None
    if mapping.get("start_at"):
        try:
            parsed = datetime.fromisoformat(mapping["start_at"])
            # Rendre la date consciente de la timezone Europe/Paris si elle est naïve
            start_date = PARIS_TZ.localize(parsed) if parsed.tzinfo is None else parsed
        except ValueError:
            logger.warning("dashboard.invalid_start_at_format", start_at=mapping["start_at"])

    # Si une date future est définie, on ne force pas le run_immediately
    should_run_now = run_immediately if not start_date else False

    _scheduler.add_job(
        job_id=mapping_id,
        sync_fn=_job,
        interval_minutes=mapping["interval_minutes"],
        run_immediately=should_run_now,
        start_date=start_date,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler, _watcher
    _watcher = NextcloudWatcher()
    _scheduler = SyncScheduler()

    # Restauration propre sans forcer l'immédiateté pour respecter les triggers natifs
    mappings = ds.get_mappings()
    for mid, m in mappings.items():
        if m.get("active", True):
            _register_job(mid, m, run_immediately=False)
    yield
    _scheduler.shutdown()


app = FastAPI(title="RAG PC4U Sync API", lifespan=lifespan)

# Montage des fichiers statiques (Assurez-vous de la cohérence de vos chemins)
static_path = Path(__file__).parent / "static"
templates_path = Path(__file__).parent / "templates"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", response_class=HTMLResponse)
async def route_index():
    html_file = templates_path / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="index.html introuvable")
    return html_file.read_text(encoding="utf-8")


@app.get("/api/status")
async def api_status():
    return {
        "nextcloud_connected": _watcher.is_connected() if _watcher else False,
        "nextcloud_url": settings.NEXTCLOUD_URL,
        "nextcloud_user": settings.NEXTCLOUD_USERNAME,
    }


@app.get("/api/mappings")
async def api_get_mappings():
    jobs = {j["id"]: j for j in _scheduler.list_jobs()}
    mappings = ds.get_mappings()
    for mid, m in mappings.items():
        m["next_run"] = jobs.get(mid, {}).get("next_run")
    return mappings


@app.post("/api/mappings", status_code=201)
async def api_create_mapping(body: MappingCreate, bg: BackgroundTasks):
    mapping = ds.add_mapping(
        remote_path=body.remote_path,
        collection_name=body.collection_name,
        interval_minutes=body.interval_minutes,
        label=body.label,
        start_at=body.start_at,
    )

    # Enregistrement du job dans le scheduler
    _register_job(mapping["id"], mapping, run_immediately=False)

    # S'il n'y a pas de planification future, exécuter directement la première fois
    if not body.start_at:
        bg.add_task(_execute_sync, mapping["id"])

    return mapping


@app.delete("/api/mappings/{mapping_id}")
async def api_delete_mapping(mapping_id: str):
    if mapping_id not in ds.get_mappings():
        raise HTTPException(status_code=404, detail="Mapping inconnu")
    if _scheduler.has_job(mapping_id):
        _scheduler._scheduler.remove_job(mapping_id)
    ds.remove_mapping(mapping_id)
    return {"status": "deleted"}


@app.post("/api/mappings/{mapping_id}/sync")
async def api_force_sync(mapping_id: str, bg: BackgroundTasks):
    if mapping_id not in ds.get_mappings():
        raise HTTPException(status_code=404, detail="Mapping inconnu")
    if _in_progress.get(mapping_id):
        return {"status": "already_running"}
    bg.add_task(_execute_sync, mapping_id)
    return {"status": "triggered"}


@app.get("/api/history")
async def api_history(limit: int = 50):
    return {
        "history": ds.get_sync_history(limit=limit),
        "stats_today": ds.get_sync_counts_today()
    }


@app.get("/api/events")
async def api_events():
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
            yield f"data: {json.dumps(payload)}\\n\\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
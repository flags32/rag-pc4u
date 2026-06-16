"""
Wrapper APScheduler pour la synchronisation périodique Nextcloud → RAG.

Un job est créé par mapping actif. Le scheduler tourne en arrière-plan
dans un thread séparé et est arrêté proprement à la fermeture de l'API.
"""

from datetime import datetime
from typing import Callable, Optional
import pytz
import structlog
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = structlog.get_logger(__name__)


class SyncScheduler:
    """
    Gère les jobs de synchronisation périodique.

    Chaque mapping (remote_path → collection) est représenté par un job
    identifié par son mapping_id. Les jobs peuvent être ajoutés, supprimés
    et déclenchés à la demande.
    """

    def __init__(self) -> None:
        self._tz = pytz.timezone("Europe/Paris")
        self._scheduler = BackgroundScheduler(
            timezone=self._tz,
            job_defaults={
                "misfire_grace_time": 120,   # tolère jusqu'à 2min de retard
                "coalesce": True,            # pas d'empilement si le job est en retard
                "max_instances": 1,          # une seule instance par job
            },
        )
        self._scheduler.start()
        logger.info("scheduler.started")

    def add_job(
        self,
        job_id: str,
        sync_fn: Callable,
        interval_minutes: int = 15,
        run_immediately: bool = True,
        start_date: Optional[datetime] = None,
    ) -> None:
        """
        Enregistre ou remplace un job de sync.
        Laisse APScheduler calculer le premier run si run_immediately est False.
        """
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)

        # Configuration dynamique des arguments pour éviter d'imposer next_run_time=None
        add_job_kwargs = {
            "trigger": IntervalTrigger(minutes=interval_minutes, start_date=start_date, timezone=self._tz),
            "id": job_id,
            "name": f"sync:{job_id}",
            "replace_existing": True,
        }

        if run_immediately:
            add_job_kwargs["next_run_time"] = datetime.now(self._tz)

        self._scheduler.add_job(sync_fn, **add_job_kwargs)

        logger.info(
            "scheduler.job_added",
            job_id=job_id,
            interval_minutes=interval_minutes,
            start_date=start_date.isoformat() if start_date else "Immédiat / Trigger par défaut",
            run_immediately=run_immediately
        )

    def trigger_immediately(self, job_id: str) -> bool:
        """Déclenche l'exécution immédiate d'un job sans modifier son planning."""
        job = self._scheduler.get_job(job_id)
        if job:
            job.modify(next_run_time=datetime.now(self._tz))
            logger.info("scheduler.triggered_immediately", job_id=job_id)
            return True
        return False

    def get_job_info(self, job_id: str) -> Optional[dict]:
        """Retourne les infos d'un job ou None s'il n'existe pas."""
        job = self._scheduler.get_job(job_id)
        if not job:
            return None
        return {\
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
        }

    def list_jobs(self) -> list[dict]:
        """Retourne la liste de tous les jobs actifs."""
        return [
            {
                "id": j.id,
                "name": j.name,
                "next_run": j.next_run_time.isoformat() if j.next_run_time else None,
            }
            for j in self._scheduler.get_jobs()
        ]

    def has_job(self, job_id: str) -> bool:
        return self._scheduler.get_job(job_id) is not None

    def shutdown(self) -> None:
        self._scheduler.shutdown(wait=False)
        logger.info("scheduler.shutdown")
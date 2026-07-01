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

PARIS_TZ = pytz.timezone("Europe/Paris")


def _localize(dt: Optional[datetime]) -> Optional[datetime]:
    """Force une date naive en Europe/Paris ; laisse passer une date déjà aware."""
    if dt is None:
        return None
    return PARIS_TZ.localize(dt) if dt.tzinfo is None else dt


class SyncScheduler:
    """
    Gère les jobs de synchronisation périodique.

    Chaque mapping (remote_path → collection) est représenté par un job
    identifié par son mapping_id. Les jobs peuvent être ajoutés, supprimés
    et déclenchés à la demande.
    """

    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler(
            timezone="Europe/Paris",
            job_defaults={
                "misfire_grace_time": 120,   # tolère jusqu'à 2min de retard
                "coalesce": True,            # pas d'empilement si le job est en retard
                "max_instances": 1,          # une seule instance par job
            },
        )
        self._scheduler.start()
        logger.info("scheduler.started")

    # Gestion des jobs

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

        Args:
            job_id           : Identifiant unique (= mapping_id).
            sync_fn          : Callable sans argument déclenché à chaque intervalle.
            interval_minutes : Fréquence de synchronisation en minutes.
            run_immediately  : Si True, la 1ère exécution est immédiate.
            start_date       : Si fourni, date/heure de la 1ère exécution planifiée.
                                Prioritaire sur run_immediately.
        """
        # Supprime l'éventuel job précédent avec le même id
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)

        start_date = _localize(start_date)

        add_job_kwargs: dict = dict(
            trigger=IntervalTrigger(
                minutes=interval_minutes,
                start_date=start_date,
                timezone=PARIS_TZ,
            ),
            id=job_id,
            name=f"sync:{job_id}",
            replace_existing=True,
        )

        # IMPORTANT : ne JAMAIS passer next_run_time=None explicitement.
        # APScheduler n'auto-calcule alors PAS le prochain run à partir du
        # trigger — le job reste sans date de déclenchement et ne se lance
        # plus jamais. Donc on n'inclut next_run_time QUE si on veut forcer
        # une valeur ; sinon on laisse le trigger décider seul.
        if start_date:
            add_job_kwargs["next_run_time"] = start_date
        elif run_immediately:
            add_job_kwargs["next_run_time"] = datetime.now(PARIS_TZ)
        # sinon (pas de start_date, run_immediately=False) : on ne passe
        # rien, IntervalTrigger calcule lui-même le 1er next_run_time.

        self._scheduler.add_job(sync_fn, **add_job_kwargs)
        logger.info(
            "scheduler.job_added",
            job_id=job_id,
            interval_minutes=interval_minutes,
            run_immediately=run_immediately,
            start_date=start_date.isoformat() if start_date else None,
        )

    def remove_job(self, job_id: str) -> bool:
        """Supprime un job. Retourne True s'il existait."""
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
            logger.info("scheduler.job_removed", job_id=job_id)
            return True
        return False

    def trigger_now(self, job_id: str) -> bool:
        """
        Déclenche l'exécution immédiate d'un job sans modifier son planning.
        Retourne False si le job n'existe pas.
        """
        job = self._scheduler.get_job(job_id)
        if job:
            # IMPORTANT : utiliser datetime.now(PARIS_TZ) et non datetime.now()
            # (naive). APScheduler compare les next_run_time en aware ; passer
            # un datetime naive provoque une TypeError silencieuse selon la
            # version, et dans tous les cas déclenche une comparaison incorrecte
            # avec le timezone du scheduler (Europe/Paris).
            job.modify(next_run_time=datetime.now(PARIS_TZ))
            logger.info("scheduler.triggered_immediately", job_id=job_id)
            return True
        return False

    def get_job_info(self, job_id: str) -> Optional[dict]:
        """Retourne les infos d'un job ou None s'il n'existe pas."""
        job = self._scheduler.get_job(job_id)
        if not job:
            return None
        return {
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

    # Lifecycle

    def shutdown(self) -> None:
        """Arrête le scheduler proprement (attend la fin des jobs en cours)."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=True)
            logger.info("scheduler.stopped")
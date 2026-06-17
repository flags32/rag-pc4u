"""
Persistance de l'état du dashboard Nextcloud.

Toutes les données (mappings configurés + historique des syncs) sont
stockées dans un seul fichier JSON. Les accès concurrents (thread du
scheduler + requêtes FastAPI) sont protégés par un verrou.

Schéma :
  {
    "mappings": {
      "<id>": {
        "id", "remote_path", "collection_name", "interval_minutes",
        "label", "created_at", "last_sync", "last_status",
        "last_stats": {"new", "modified", "deleted", "errors"},
        "active"
      }
    },
    "sync_history": [
      { "id", "mapping_id", "timestamp", "status", "new", "modified",
        "deleted", "errors", "started_at", "finished_at",
        "error_message" (optionnel) }
    ]   ← trié du plus récent au plus ancien, limité à 500 entrées
  }
"""

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from rag_pc4u.core.tz_utils import now_paris_naive

STATE_FILE = Path(__file__).parent / "mapping/dashboard_state.json"
_lock = threading.Lock()

# Création du dossier au chargement du module — garanti avant tout accès I/O.
# parents=True : crée /app/src/rag_pc4u/dashboard/mapping/ en une seule fois.
# exist_ok=True : silencieux si le dossier existe déjà (restart du conteneur).
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


# I/O

def _load() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            pass
    return {"mappings": {}, "sync_history": []}


def _save(state: dict) -> None:
    # Sécurité supplémentaire : recrée le dossier s'il a été supprimé à chaud
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# Mappings

def get_mappings() -> dict:
    """Retourne une copie du dict de mappings."""
    with _lock:
        return dict(_load()["mappings"])


def get_mapping(mapping_id: str) -> Optional[dict]:
    with _lock:
        return _load()["mappings"].get(mapping_id)


def add_mapping(
        remote_path: str,
        collection_name: str,
        interval_minutes: int = 15,
        label: Optional[str] = None,
        start_at: Optional[str] = None,
) -> dict:
    """
    Crée un nouveau mapping et le persiste.

    Args:
        start_at : Date/heure ISO de la 1ère sync planifiée (optionnel).

    Returns:
        Le mapping créé avec son id généré.
    """
    with _lock:
        state = _load()
        mid = str(uuid.uuid4())[:8]
        label = label or f"{remote_path.rstrip('/')}  →  {collection_name}"
        mapping = {
            "id": mid,
            "remote_path": remote_path,
            "collection_name": collection_name,
            "interval_minutes": interval_minutes,
            "start_at": start_at,
            "label": label,
            "created_at": now_paris_naive().isoformat(),
            "last_sync": None,
            "last_status": None,
            "last_stats": None,
            "active": True,
        }
        state["mappings"][mid] = mapping
        _save(state)
        return mapping


def delete_mapping(mapping_id: str) -> bool:
    """Supprime un mapping. Retourne True s'il existait."""
    with _lock:
        state = _load()
        if mapping_id not in state["mappings"]:
            return False
        del state["mappings"][mapping_id]
        _save(state)
        return True


def update_mapping_after_sync(mapping_id: str, sync_stats: dict) -> None:
    """
    Met à jour last_sync, last_status et last_stats d'un mapping
    après une synchronisation (succès ou erreur).
    """
    with _lock:
        state = _load()
        if mapping_id not in state["mappings"]:
            return
        m = state["mappings"][mapping_id]
        m["last_sync"] = now_paris_naive().isoformat()
        m["last_status"] = sync_stats.get("status")
        m["last_stats"] = {
            k: sync_stats.get(k, 0)
            for k in ("new", "modified", "deleted", "errors")
        }
        _save(state)


# Historique des syncs

def add_sync_record(mapping_id: str, stats: dict) -> None:
    """
    Insère un enregistrement de sync en tête de l'historique.
    Limite l'historique à 500 entrées.
    """
    with _lock:
        state = _load()
        record = {
            "id": str(uuid.uuid4())[:8],
            "mapping_id": mapping_id,
            "timestamp": now_paris_naive().isoformat(),
            **stats,
        }
        state["sync_history"].insert(0, record)
        state["sync_history"] = state["sync_history"][:500]
        _save(state)


def get_sync_history(
        limit: int = 50,
        mapping_id: Optional[str] = None,
) -> list:
    """
    Retourne l'historique des syncs, optionnellement filtré par mapping.

    Args:
        limit      : Nombre maximum d'entrées retournées.
        mapping_id : Si fourni, filtre sur ce mapping uniquement.
    """
    with _lock:
        state = _load()
        history = state["sync_history"]
        if mapping_id:
            history = [h for h in history if h.get("mapping_id") == mapping_id]
        return history[:limit]


def get_sync_counts_today() -> dict:
    """
    Retourne le nombre de syncs et d'erreurs pour aujourd'hui.
    Utilisé par le dashboard pour les stats en temps réel.
    """
    today = now_paris_naive().strftime("%Y-%m-%d")
    with _lock:
        state = _load()
        history = state["sync_history"]

    total = sum(1 for h in history if (h.get("timestamp") or "").startswith(today))
    errors = sum(
        1 for h in history
        if (h.get("timestamp") or "").startswith(today) and h.get("status") == "error"
    )
    return {"total": total, "errors": errors}
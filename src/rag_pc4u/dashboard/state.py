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

STATE_FILE = Path(__file__).parent / "mapping/dashboard_state.json"
_lock = threading.Lock()

# Création du dossier au chargement du module — garanti avant tout accès I/O.
# parents=True : crée /app/src/rag_pc4u/dashboard/mapping/ en une seule fois.
# exist_ok=True : silencieux si le dossier existe déjà (restart du conteneur).
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load() -> dict:
    if not STATE_FILE.exists():
        return {"mappings": {}, "sync_history": []}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"mappings": {}, "sync_history": []}


def _save(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_mappings() -> dict:
    with _lock:
        return _load()["mappings"]


def add_mapping(
    remote_path: str,
    collection_name: str,
    interval_minutes: int = 15,
    label: Optional[str] = None,
    start_at: Optional[str] = None,
) -> dict:
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
            "created_at": datetime.now().isoformat(),
            "last_sync": None,
            "last_status": None,
            "last_stats": None,
            "active": True,
        }
        state["mappings"][mid] = mapping
        _save(state)
        return mapping


def remove_mapping(mapping_id: str) -> None:
    with _lock:
        state = _load()
        if mapping_id in state["mappings"]:
            del state["mappings"][mapping_id]
            state["sync_history"] = [h for h in state["sync_history"] if h.get("mapping_id") != mapping_id]
            _save(state)


def update_mapping_sync_status(mapping_id: str, status: str, stats: Optional[dict] = None) -> None:
    with _lock:
        state = _load()
        if mapping_id in state["mappings"]:
            state["mappings"][mapping_id]["last_sync"] = datetime.now().isoformat()
            state["mappings"][mapping_id]["last_status"] = status
            if stats:
                state["mappings"][mapping_id]["last_stats"] = stats
            _save(state)


def add_sync_history_record(mapping_id: str, stats: dict) -> None:
    with _lock:
        state = _load()
        record = {
            "id": str(uuid.uuid4())[:8],
            "mapping_id": mapping_id,
            "timestamp": datetime.now().isoformat(),
            **stats,
        }
        state["sync_history"].insert(0, record)
        state["sync_history"] = state["sync_history"][:500]
        _save(state)


def get_sync_history(limit: int = 50, mapping_id: Optional[str] = None) -> list:
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
    today = datetime.now().strftime("%Y-%m-%d")
    with _lock:
        state = _load()
        history = state["sync_history"]

    total = sum(1 for h in history if h.get("timestamp", "").startswith(today))
    errors = sum(1 for h in history if h.get("timestamp", "").startswith(today) and h.get("status") == "error")
    return {"total": total, "errors": errors}
"""
Persistance de l'état du dashboard Nextcloud.

Toutes les données (mappings configurés + historique des syncs) sont
stockées dans un seul fichier JSON. Les accès concurrents (thread du
scheduler + requêtes FastAPI) sont protégés par un verrou.

Schéma :
  {
    "mappings": {
      "<id>": {
        "id", "remote_paths", "collection_name", "interval_minutes",
        "label", "created_at", "last_sync", "last_status",
        "last_stats": {"new", "modified", "deleted", "errors"},
        "pending_files", "active"
      }
    },
    "sync_history": [
      { "id", "mapping_id", "timestamp", "status", "new", "modified",
        "deleted", "errors", "started_at", "finished_at",
        "error_message" (optionnel) }
    ]   ← trié du plus récent au plus ancien, limité à 500 entrées
  }

Changements v2 :
  - remote_path (str)  →  remote_paths (list[str])  — point 2 : multi-chemins
    Rétrocompatibilité : les anciens enregistrements avec "remote_path" str
    sont normalisés automatiquement à la lecture (_normalize_mapping).
  - Unicité du label par mapping actif  — point 1
  - Unicité collection → mapping actif  — point 1 (une collection ne peut
    appartenir qu'à un seul mapping actif à la fois)
  - Recherche par label ou collection   — point 1
"""

import json
import threading
import uuid
from pathlib import Path
from typing import Optional

from rag_pc4u.core.tz_utils import now_paris_naive

STATE_FILE = Path(__file__).parent / "mapping/dashboard_state.json"
_lock = threading.Lock()

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
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _normalize_mapping(m: dict) -> dict:
    """
    Rétrocompatibilité : migre les anciens enregistrements qui utilisaient
    encore remote_path (str) vers remote_paths (list[str]).
    Non destructif — ne réécrit pas le JSON sur disque, opère uniquement
    en mémoire pour ne pas créer de writes inutiles à chaque lecture.
    """
    if "remote_path" in m and "remote_paths" not in m:
        m = dict(m)
        m["remote_paths"] = [m.pop("remote_path")]
    return m


# Helpers de validation (point 1)

def _label_exists(state: dict, label: str, exclude_id: Optional[str] = None) -> bool:
    """Vérifie si un label est déjà utilisé par un mapping actif."""
    for mid, m in state["mappings"].items():
        if mid == exclude_id:
            continue
        if m.get("active") and m.get("label", "").strip().lower() == label.strip().lower():
            return True
    return False


def _collection_used(state: dict, collection_name: str, exclude_id: Optional[str] = None) -> bool:
    """
    Vérifie si une collection est déjà associée à un mapping actif.
    Une collection ne peut appartenir qu'à un seul mapping actif.
    """
    for mid, m in state["mappings"].items():
        if mid == exclude_id:
            continue
        if m.get("active") and m.get("collection_name") == collection_name:
            return True
    return False


def _path_conflict(
    state: dict,
    remote_paths: list[str],
    exclude_id: Optional[str] = None,
) -> Optional[str]:
    """
    Vérifie qu'aucun des chemins demandés n'est déjà utilisé par un autre
    mapping actif. Un chemin physique Nextcloud ne peut appartenir qu'à un
    seul mapping à la fois (contrainte métier : un fichier = un seul mapping).

    Retourne une string d'erreur descriptive si conflit, None sinon.
    """
    for mid, m in state["mappings"].items():
        if mid == exclude_id or not m.get("active"):
            continue
        existing_paths = set(
            _normalize_mapping(m).get("remote_paths", [])
        )
        conflicts = set(remote_paths) & existing_paths
        if conflicts:
            conflict_list = ", ".join(f"« {p} »" for p in sorted(conflicts))
            return (
                f"Le(s) chemin(s) {conflict_list} "
                f"sont déjà utilisés par le mapping « {m.get('label', mid)} »."
            )
    return None


# Mappings

def get_mappings() -> dict:
    """Retourne une copie du dict de mappings (normalisés)."""
    with _lock:
        raw = _load()["mappings"]
        return {mid: _normalize_mapping(m) for mid, m in raw.items()}


def get_mapping(mapping_id: str) -> Optional[dict]:
    with _lock:
        m = _load()["mappings"].get(mapping_id)
        return _normalize_mapping(m) if m else None


def search_mappings(
    label: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> list[dict]:
    """
    Recherche de mappings par label (sous-chaîne, insensible à la casse)
    ou par collection exacte. Les deux filtres sont cumulatifs (AND).

    Point 1 du rapport : permettre la recherche d'un mapping par collection
    Qdrant ou par libellé/nom.
    """
    with _lock:
        mappings = _load()["mappings"]

    results = []
    for m in mappings.values():
        m = _normalize_mapping(m)
        if label and label.strip().lower() not in m.get("label", "").lower():
            continue
        if collection_name and m.get("collection_name") != collection_name:
            continue
        results.append(m)
    return results


def add_mapping(
        remote_paths: list[str],
        collection_name: str,
        interval_minutes: int = 15,
        label: Optional[str] = None,
        start_at: Optional[str] = None,
) -> dict | str:
    """
    Crée un nouveau mapping et le persiste.

    Returns:
        Le mapping créé (dict), ou une string d'erreur si une contrainte
        d'unicité est violée (label déjà utilisé, ou collection déjà liée
        à un autre mapping actif).

    Point 1 : unicité du label et de la collection par mapping actif.
    Point 2 : remote_paths est une liste (dossiers ET/OU fichiers individuels).
    """
    with _lock:
        state = _load()
        mid = str(uuid.uuid4())[:8]

        # Label par défaut basé sur le premier chemin si non fourni
        resolved_label = label or f"{remote_paths[0].rstrip('/')}  →  {collection_name}"

        # Contrôles d'unicité (point 1)
        if _label_exists(state, resolved_label):
            return f"Un mapping actif utilise déjà le libellé « {resolved_label} »."
        if _collection_used(state, collection_name):
            return (
                f"La collection « {collection_name} » est déjà associée à un "
                f"mapping actif. Une collection ne peut appartenir qu'à un seul mapping."
            )
        path_err = _path_conflict(state, remote_paths)
        if path_err:
            return path_err

        mapping = {
            "id": mid,
            "remote_paths": remote_paths,     # list[str] — point 2
            "collection_name": collection_name,
            "interval_minutes": interval_minutes,
            "start_at": start_at,
            "label": resolved_label,
            "created_at": now_paris_naive().isoformat(),
            "last_sync": None,
            "last_status": None,
            "last_stats": None,
            "pending_files": [],
            "active": True,
        }
        state["mappings"][mid] = mapping
        _save(state)
        return mapping


def update_mapping(
        mapping_id: str,
        remote_paths: Optional[list[str]] = None,
        collection_name: Optional[str] = None,
        interval_minutes: Optional[int] = None,
        label: Optional[str] = None,
        start_at: Optional[str] = None,
) -> dict | str | None:
    """
    Met à jour les champs fournis d'un mapping existant.
    Retourne le mapping mis à jour, une string d'erreur si contrainte
    d'unicité violée, ou None si le mapping n'existe pas.
    """
    with _lock:
        state = _load()
        if mapping_id not in state["mappings"]:
            return None
        m = state["mappings"][mapping_id]

        target_label = label if label is not None else m.get("label")
        target_collection = collection_name if collection_name is not None else m.get("collection_name")

        # Contrôles d'unicité (point 1)
        if label is not None and _label_exists(state, label, exclude_id=mapping_id):
            return f"Un mapping actif utilise déjà le libellé « {label} »."
        if collection_name is not None and _collection_used(state, collection_name, exclude_id=mapping_id):
            return (
                f"La collection « {collection_name} » est déjà associée à un "
                f"mapping actif. Une collection ne peut appartenir qu'à un seul mapping."
            )
        if remote_paths is not None:
            path_err = _path_conflict(state, remote_paths, exclude_id=mapping_id)
            if path_err:
                return path_err

        if remote_paths is not None:
            m["remote_paths"] = remote_paths
        if collection_name is not None:
            m["collection_name"] = collection_name
        if interval_minutes is not None:
            m["interval_minutes"] = interval_minutes
        if label is not None:
            m["label"] = label
        # start_at toujours réécrit (même à None) pour permettre d'effacer
        # une planification existante.
        m["start_at"] = start_at
        _save(state)
        return _normalize_mapping(m)


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
        m["pending_files"] = sync_stats.get("pending_files", [])
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
    with _lock:
        state = _load()
        history = state["sync_history"]
        if mapping_id:
            history = [h for h in history if h.get("mapping_id") == mapping_id]
        return history[:limit]


def get_sync_counts_today() -> dict:
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
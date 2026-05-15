from typing import Any

CLIENT_ID_FIELD = "meta.client_id"


def filter_for(client_id: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Filtre Haystack qui force le cloisonnement par client_id.

    Combine en AND avec `extra` si fourni. Sans `extra`, retourne un filtre plat.
    """
    base = {"field": CLIENT_ID_FIELD, "operator": "==", "value": client_id}

    if extra is None:
        return base

    extra_conditions = (
        extra["conditions"]
        if extra.get("operator") == "AND" and "conditions" in extra
        else [extra]
    )

    return {"operator": "AND", "conditions": [base, *extra_conditions]}

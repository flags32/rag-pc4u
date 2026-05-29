import logging

import structlog

from rag_pc4u.core.config import settings


def configure_logging() -> None:
    """Configure le logging structuré. Applique le fail-fast sur le niveau."""
    #  secure_logging_config() n'était jamais appelée,
    # la validation était purement décorative. On l'intègre directement ici.
    level_str = settings.log_level.upper()
    if not validate_log_level(level_str):
        raise ValueError(
            f"Niveau de log invalide : '{level_str}'. "
            f"Valeurs acceptées : DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )

    level = logging.getLevelName(level_str)
    logging.basicConfig(format="%(message)s", level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def validate_log_level(level: str) -> bool:
    """Vérifie si le niveau de log est valide."""
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    return level.upper() in valid_levels


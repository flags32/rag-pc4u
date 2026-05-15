import logging

import structlog

from rag_pc4u.core.config import settings


def configure_logging() -> None:
    level = logging.getLevelName(settings.log_level.upper())
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
#Fonction de sécurité juste en dessous avec pour but de faire en sorte que les logging soit bien gérés
def secure_logging_config():
    """
    Assure que la configuration des logs est sécurisée et conforme aux normes de sécurité.
    """
    s = settings.log_level.upper()
    if not validate_log_level(s):
        raise ValueError(f"Invalid log level: {s}")#le if est la pour vérifier que le niveau de log est valide

def validate_log_level(level: str) -> bool:
    """
    Vérifie si le niveau de log est valide en fonction des normes de sécurité.
    """
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    return level.upper() in valid_levels

"""on appelle le tout le principe du Fail-fast"""

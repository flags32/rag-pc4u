"""Run module Rag PC4U"""

import structlog
from pathlib import Path
from rag_pc4u.core.logging import configure_logging
from rag_pc4u.ingestion.pipeline import build_indexing_pipeline

logger = structlog.get_logger(__name__)


def test_ingestion():
    configure_logging()

    # 1. ingestion de fichier test
    logger.info("Démarrage du pipeline d'ingestion...")
    pipeline = build_indexing_pipeline()

    results = pipeline.run({
        "converter": {"sources": ["/home/user/Documents/projet rag/tests/test1",
                                  "/home/user/Documents/projet rag/tests/test2",
                                  "/home/user/Documents/projet rag/tests/test3"]},
        "enricher": {"client_id": "client_demo"}
    })

    logger.info("Ingestion terminée", documents_ecrits=results["writer"]["documents_written"])



if __name__ == "__main__":
    test_ingestion()
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
        "converter": {"sources": ["/home/user/Documents/projet_rag/tests/test1",
                                  "/home/user/Documents/projet_rag/tests/test2",
                                  "/home/user/Documents/projet_rag/tests/test3",
                                  "/home/user/Documents/projet_rag/tests/test4",
                                  "/home/user/Documents/projet_rag/tests/test5",
                                  "/home/user/Documents/projet_rag/tests/test6",
                                  "/home/user/Documents/projet_rag/tests/test7",
                                  "/home/user/Documents/projet_rag/tests/test8",
                                  "/home/user/Documents/projet_rag/tests/test9",
                                  "/home/user/Documents/projet_rag/tests/test10",
                                  "/home/user/Documents/projet_rag/tests/test11",
                                  "/home/user/Documents/projet_rag/tests/test12",
                                  "/home/user/Documents/projet_rag/tests/test14",
                                  "/home/user/Documents/projet_rag/tests/test15",
                                  "/home/user/Documents/projet_rag/tests/test16",
                                  ]},
        "enricher": {"client_id": "client_demo"}
    })

    logger.info("Ingestion terminée", documents_ecrits=results["writer"]["documents_written"])



if __name__ == "__main__":
    test_ingestion()
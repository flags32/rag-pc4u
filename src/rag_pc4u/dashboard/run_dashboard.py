"""
Point d'entrée du dashboard Nextcloud → RAG.

Deux façons de lancer :

  # Avec uvicorn directement (recommandé en production) :
  uvicorn rag_pc4u.dashboard.api:app --host 0.0.0.0 --port 8001

  # Ou via ce script :
  python -m rag_pc4u.dashboard.run_dashboard
"""

import uvicorn

from rag_pc4u.core.logger_config import configure_logging


def main() -> None:
    configure_logging()
    uvicorn.run(
        "rag_pc4u.dashboard.api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
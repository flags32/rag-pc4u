from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, List, Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import des composants Haystack / Qdrant et configurations
from rag_pc4u.retrieval.pipeline import build_hybrid_rag_pipeline
from rag_pc4u.core.config import settings
from rag_pc4u.core.logging import configure_logging
from rag_pc4u.retrieval.services import answer

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    app.state.query_pipeline = build_hybrid_rag_pipeline()
    logger.info(
        "api.startup",
        client_id=settings.client_id,
        collection=settings.collection_name,
        qdrant=settings.qdrant_url,
        ollama=settings.ollama_host,
    )
    yield
    logger.info("api.shutdown")


app = FastAPI(
    title="RAG PC4U API",
    version="0.1.0",
    description="API RAG on-premise souveraine — stack Haystack + Qdrant + Ollama",
    lifespan=lifespan,
)

# Configuration CORS indispensable pour qu'Open WebUI puisse requêter l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production à l'URL de l'Open WebUI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- MODÈLES PYDANTIC POUR LA COMPATIBILITÉ OPENAI (OPEN WEBUI) ---

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-rag-pc4u"
    object: str = "chat.completion"
    created: int = 1710000000
    model: str
    choices: List[ChatCompletionResponseChoice]


# --- ENDPOINTS ---

@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "client_id": settings.client_id,
        "collection": settings.collection_name,
    }


@app.get("/v1/models")
async def list_models():
    """
    Endpoint requis par Open WebUI pour détecter les modèles disponibles.
    On expose le RAG Haystack comme un modèle sémantique standard.
    """
    return {#sert a retourner une list de model
        "object": "list",
        "data": [
            {
                "id": "rag-hybrid-pc4u",
                "object": "model",
                "created": 1710000000,
                "owned_by": "pc4u"
            }
        ]
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
        request: Request, # <--- Récupère l'objet Request de FastAPI
        completion_request: ChatCompletionRequest, # renommé pour clarté
        x_client_id: Optional[str] = Header(None, alias="X-Client-Id")
):
    """
    Endpoint OpenAI simulé qui reçoit les messages d'Open WebUI,
    extrait la question, exécute le pipeline Haystack/Qdrant et renvoie la réponse.
    """
    # 1. Extraction du dernier message de l'utilisateur
    user_messages = [msg.content for msg in completion_request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="Aucun message utilisateur trouvé.")

    last_question = user_messages[-1]

    # 2. Gestion du multi-tenant (Cloisonnement client)
    # Reçoit le client_id depuis les headers, sinon se replie sur la config de démo
    client_id = x_client_id or settings.client_id

    try:
        pipeline = request.app.state.query_pipeline
        logger.info("api.rag.execute", client_id=client_id, model_requested=completion_request.model)

        # 3. Appel de ton pipeline Haystack (Dense + Sparse + Qdrant + Ollama)
        rag_response = answer(query=last_question, client_id=client_id, pipeline=pipeline)

        # 4. Formatage du retour au format OpenAI attendu par Open WebUI
        return ChatCompletionResponse(
            model=completion_request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=rag_response.answer)
                )
            ]
        )
    except Exception as e:
        logger.error("api.rag.error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Erreur interne du RAG Haystack: {str(e)}")


def run() -> None:
    uvicorn.run(
        "rag_pc4u.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()













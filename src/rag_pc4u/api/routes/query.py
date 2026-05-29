"""Module de routes pour les requêtes de l'API RAG PC4U."""

from fastapi import APIRouter
import structlog
from haystack import Pipeline

# On utilise nos nouveaux schémas et dépendances dédiés à l'API
from rag_pc4u.api.schemas import HTTPQueryRequest, HTTPQueryResponse
#from rag_pc4u.api.dependencies import get_client_id_from_key, get_query_pipeline
from rag_pc4u.core.config import settings
from rag_pc4u.retrieval.services import answer
from pydantic import BaseModel
from typing import Any, List, Optional
import time
from fastapi import HTTPException, Header, Request

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Query"])


# --- MODELES PYDANTIC (sortis de main.py) ---

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
    created: int = 0
    model: str
    choices: List[ChatCompletionResponseChoice]

    def __init__(self, **data):
        if "created" not in data or data["created"] == 0:
            data["created"] = int(time.time())
        super().__init__(**data)


# --- ENDPOINTS ---

@router.get("/v1/models")
async def list_models():
    """Endpoint requis par Open WebUI pour detecter les modeles disponibles."""
    return {
        "object": "list",
        "data": [
            {
                "id": "rag-hybrid-pc4u",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "pc4u",
            }
        ],
    }


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
        request: Request,
        completion_request: ChatCompletionRequest,
        x_client_id: Optional[str] = Header(None, alias="X-Client-Id"),
):
    """
    Endpoint OpenAI simule pour Open WebUI.
    Extrait la question, execute le pipeline RAG, renvoie la reponse.
    """
    user_messages = [
        msg.content for msg in completion_request.messages if msg.role == "user"
    ]
    if not user_messages:
        raise HTTPException(status_code=400, detail="Aucun message utilisateur trouve.")

    last_question = user_messages[-1]
    client_id = x_client_id if x_client_id is not None else settings.client_id

    try:
        pipeline = request.app.state.query_pipeline
        logger.info(
            "api.rag.execute",
            client_id=client_id,
            model_requested=completion_request.model,
        )

        rag_response = answer(query=last_question, client_id=client_id, pipeline=pipeline)

        return ChatCompletionResponse(
            model=completion_request.model,
            choices=[
                ChatCompletionResponseChoice(
                    message=ChatMessage(role="assistant", content=rag_response.answer)
                )
            ],
        )
    except Exception as e:
        logger.error("api.rag.error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne du RAG Haystack: {str(e)}",
        )

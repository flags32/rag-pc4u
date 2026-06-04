"""Routes de requêtes RAG PC4U — compatibilité OpenAI pour Open WebUI."""
import time
import structlog
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

from rag_pc4u.core.config import settings
from rag_pc4u.retrieval.services import answer

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Query"])


# ── Schémas OpenAI ────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/v1/models")
async def list_models():
    """Endpoint requis par Open WebUI. Expose les collections comme des modèles."""
    models_list = []

    # On génère un "modèle" pour chaque collection connue
    for collection in settings.List_collection:
        models_list.append({
            "id": collection,  # C'est ce qui s'affichera dans le sélecteur Open WebUI
            "object": "model",
            "created": int(time.time()),
            "owned_by": "pc4u",
        })

    return {
        "object": "list",
        "data": models_list if models_list else [
            {
                "id": settings.default_collection,  # Fallback de sécurité
                "object": "model",
                "created": int(time.time()),
                "owned_by": "pc4u",
            }
        ],
    }


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
        completion_request: ChatCompletionRequest,
        x_collection: Optional[str] = Header(None, alias="X-Collection"),
):
    """
    Endpoint compatible OpenAI pour Open WebUI.

    Header X-Collection : nom de la collection Qdrant à interroger.
                          Si absent, utilise settings.default_collection.

    La route fait exactement 3 choses :
    1. Extraire la dernière question utilisateur.
    2. Déterminer la collection cible.
    3. Appeler answer() et retourner la réponse.

    Tout le reste (pipeline, format Haystack, LLM) est dans retrieval/services.py.
    """
    user_messages = [
        msg.content for msg in completion_request.messages if msg.role == "user"
    ]
    if not user_messages:
        raise HTTPException(status_code=400, detail="Aucun message utilisateur trouvé.")

    last_question = user_messages[-1]
    collection_name = completion_request.model

    # Vérification de sécurité optionnelle
    if collection_name not in settings.List_collection:
        logger.warning(f"Collection {collection_name} inconnue, fallback sur default.")
        collection_name = settings.default_collection

    try:
        logger.info(
            "api.rag.execute",
            collection=collection_name,
            model_requested=completion_request.model,
        )

        rag_response = answer(query=last_question, collection_name=collection_name)

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
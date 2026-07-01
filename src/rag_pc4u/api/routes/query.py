"""Routes de requêtes RAG PC4U — compatibilité OpenAI pour Open WebUI."""
import json
import time
import structlog
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_pc4u.core.config import settings
from rag_pc4u.retrieval.services import answer, answer_stream

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Query"])


# Schémas OpenAI

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


# Endpoints

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


async def _stream_openai_format(query: str, collection_name: str, model: str):
    """Convertit answer_stream() en format SSE compatible OpenAI."""
    chunk_id = "chatcmpl-rag-pc4u-stream"
    created = int(time.time())

    try:
        async for token in answer_stream(query=query, collection_name=collection_name):
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(payload)}\n\n"

        final_payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(final_payload)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error("api.rag.stream_error", error=str(e))
        error_payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n\n[Erreur RAG : {str(e)}]"},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"


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
    keywords = ["dashboard"]

    keywordsingestsion = ["ingér", "ingest", "indexer"]

    if any(kw in last_question.lower() for kw in keywordsingestsion):
        logger.info("api.rag.ingestion_requested", question=last_question)

        artifact_response = ("""si vous êtes sûr de vouloir ingérer, appuyez sur le petit bouton :
    ```xml
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" width="24" height="24">
      <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z" />
    </svg>
    ```""")

        return ChatCompletionResponse(
            model=completion_request.model,
            choices=[
                ChatCompletionResponseChoice(
                    message=ChatMessage(role="assistant", content=artifact_response)
                )
            ],
        )

    elif any(kw in last_question.lower() for kw in keywords):
        logger.info("api.rag.dashboard_requested", question=last_question)

        dashboard_url = "http://192.168.204.23:8001/"

        artifact_response = (
            f"🔗 **[Ouvrir le dashboard Nextcloud → RAG]({dashboard_url})**"
        )

        return ChatCompletionResponse(
            model=completion_request.model,
            choices=[
                ChatCompletionResponseChoice(
                    message=ChatMessage(role="assistant", content=artifact_response)
                )
            ],
        )

    # Reste de ton code initial (Appel à Haystack)
    collection_name = completion_request.model
    #verification optionnel
    if collection_name not in settings.List_collection:
        logger.warning(f"Collection {collection_name} inconnue, fallback sur default.")
        collection_name = settings.default_collection

    # Branche streaming : si demandé, on bascule en SSE avant tout appel à answer()
    if completion_request.stream:
        logger.info(
            "api.rag.execute_stream",
            collection=collection_name,
            model_requested=completion_request.model,
        )
        return StreamingResponse(
            _stream_openai_format(last_question, collection_name, completion_request.model),
            media_type="text/event-stream",
        )

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
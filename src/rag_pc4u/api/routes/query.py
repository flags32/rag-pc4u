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
    if any(kw in last_question.lower() for kw in keywords):
        logger.info("api.rag.dashboard_requested", question=last_question)

        dashboard_url = "http://192.168.204.23:8001/"

        artifact_response = (
            f"Voici le lien vers votre panneau de gestion des synchronisations Nextcloud :\n\n"
            f"🔗 **[Ouvrir le dashboard Nextcloud → RAG]({dashboard_url})**\n\n"
            f"Vous pouvez également cliquer sur le bouton ci-dessous pour l'ouvrir dans un nouvel onglet.\n\n"
            "```html\n"
            f'<!DOCTYPE html><html><head><meta charset="UTF-8">'
            f'<style>'
            f'body{{margin:0;display:flex;align-items:center;justify-content:center;'
            f'min-height:100vh;background:#07090f;font-family:Outfit,sans-serif;}}'
            f'.card{{background:#101520;border:1px solid #1c2535;border-radius:12px;'
            f'padding:2.5rem 3rem;text-align:center;max-width:420px;}}'
            f'h2{{color:#e8eef8;margin-bottom:.5rem;font-size:1.3rem;}}'
            f'p{{color:#8a9ab8;font-size:.9rem;margin-bottom:1.8rem;}}'
            f'a{{display:inline-block;background:#00e5c3;color:#07090f;'
            f'font-weight:700;padding:.85rem 2rem;border-radius:8px;'
            f'text-decoration:none;font-size:1rem;letter-spacing:.02em;}}'
            f'a:hover{{background:#00c9aa;}}'
            f'</style></head><body>'
            f'<div class="card">'
            f'<h2>⬡ RAG PC4U — Sync Dashboard</h2>'
            f'<p>Panneau de gestion des synchronisations Nextcloud → Collections RAG</p>'
            f'<a href="http://192.168.204.23:8001/">Ouvrir le Dashboard →</a>'
            f'</div></body></html>\n'
            "```"
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
import io
import os
from pathlib import Path
from typing import Any, List, Union

from haystack import default_to_dict, default_from_dict, Document
from haystack.core.component import component
from haystack.dataclasses import ByteStream
from haystack.utils import Secret  # CORRECTION : Import depuis Haystack (Starlette n'a pas .resolve_value())
from haystack.utils.http_client import init_http_client
from openai import OpenAI

from rag_pc4u.core.components import logger


@component
class RemoteWhisperTranscriber:
    """
    Composant Haystack qui envoie l'audio au conteneur Docker Whisper
    sans surcharger la RAM ou le CPU de l'API Rag.
    """

    def __init__(
            self,
            # CORRECTION : Sécurité locale si la variable OPENAI_API_KEY n'est pas définie dans le .env
            api_key: Secret = Secret.from_token(os.environ.get("OPENAI_API_KEY", "local-dummy-key")),
            # CORRECTION : Modèle par défaut aligné sur ton serveur local
            model: str = "large-v3",
            # OPTIMISATION : URL par défaut pointant directement sur ton service Docker
            api_base_url: str | None = "http://rag-whisper:8002/v1",
            organization: str | None = None,
            http_client_kwargs: dict[str, Any] | None = None,
            **kwargs: Any,
    ) -> None:
        self.organization = organization
        self.model = model
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.http_client_kwargs = http_client_kwargs

        # Only response_format = "json" is supported
        whisper_params = kwargs
        response_format = whisper_params.get("response_format", "json")
        if response_format != "json":
            logger.warning(
                "RemoteWhisperTranscriber only supports 'response_format: json'. This parameter will be overwritten."
            )
        whisper_params["response_format"] = "json"
        self.whisper_params = whisper_params

        self.client = OpenAI(
            api_key=api_key.resolve_value(),
            organization=organization,
            base_url=api_base_url,
            http_client=init_http_client(self.http_client_kwargs, async_client=False),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.
        """
        return default_to_dict(
            self,
            api_key=self.api_key,
            model=self.model,
            organization=self.organization,
            api_base_url=self.api_base_url,
            http_client_kwargs=self.http_client_kwargs,
            **self.whisper_params,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RemoteWhisperTranscriber":
        """
        Deserializes the component from a dictionary.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]]) -> dict[str, Any]:
        """
        Transcribes the list of audio files into a list of documents.
        """
        documents = []

        for source in sources:
            if not isinstance(source, ByteStream):
                path = source
                source = ByteStream.from_file_path(Path(source))
                source.meta["file_path"] = str(path)

            file = io.BytesIO(source.data)
            file.name = str(source.meta["file_path"]) if "file_path" in source.meta else "__fallback__.wav"

            # L'appel réseau part vers ton conteneur sans bloquer ton CPU local
            content = self.client.audio.transcriptions.create(
                file=file,
                model=self.model,
                **self.whisper_params
            )

            doc = Document(content=content.text, meta=source.meta)
            documents.append(doc)

        return {"documents": documents}
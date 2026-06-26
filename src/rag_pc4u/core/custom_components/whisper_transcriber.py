"""
Composant Haystack custom pour la transcription audio via un serveur Whisper distant.

FIXES appliqués :
  1. Timeout httpx configuré explicitement (défaut 7200s = 2h).
     Le client OpenAI avait un timeout par défaut de 600s, ce qui causait
     des retries prématurés sur des fichiers audio longs, aboutissant à une
     cascade de requêtes en parallèle sur un serveur avec WHISPER_NUM_WORKERS=1.
  2. max_retries=0 : on désactive les retries automatiques du client OpenAI.
     Un retry sur une transcription longue soumet un doublon au serveur
     pendant que celui-ci traite encore la première requête.
     La gestion des erreurs est confiée à run.py (IngestionPendingFilesError).
"""

import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Union

import httpx
import structlog
from haystack import Document, component
from haystack.dataclasses import ByteStream
from openai import OpenAI

logger = structlog.get_logger(__name__)

# Timeout par défaut : 2 heures.
# Justification : large-v3 sur CPU en float32 traite environ 5–10× le temps réel.
# Un audio de 30 min peut prendre jusqu'à 3h dans le pire des cas.
# On préfère un timeout long + max_retries=0 à un timeout court + retries.
_DEFAULT_TIMEOUT_SECONDS = 7_200.0  # 2 heures

# Extensions audio/vidéo reconnues
_AUDIO_VIDEO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".flac",
    ".mp4", ".mov", ".mkv", ".avi", ".webm",
}


def _resolve_mime(path: Path) -> str:
    """Détermine un MIME type à partir de l'extension, avec fallback."""
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


@component
class RemoteWhisperTranscriber:
    """
    Composant Haystack qui envoie des fichiers audio/vidéo à un serveur
    Whisper distant (compatible OpenAI /v1/audio/transcriptions) et retourne
    les transcriptions sous forme de Documents Haystack.

    Args:
        api_base_url    : URL de base du serveur Whisper. Ex : "http://rag-whisper:8000/v1"
        model           : Nom du modèle Whisper déclaré côté serveur.
                          Utilisé uniquement comme paramètre d'API (le serveur
                          peut l'ignorer s'il n'expose qu'un seul modèle).
        language        : Code BCP-47 optionnel pour forcer la langue (ex : "fr").
                          Si None, Whisper détecte automatiquement.
        timeout_seconds : Timeout HTTP en secondes pour la phase de lecture
                          (= durée maximale de transcription tolérée).
                          Défaut : 7 200 s (2 heures).
    """

    def __init__(
        self,
        api_base_url: str,
        model: str = "Systran/faster-whisper-large-v3",
        language: Optional[str] = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ):
        self.api_base_url = api_base_url
        self.model = model
        self.language = language
        self.timeout_seconds = timeout_seconds

    def _build_client(self) -> OpenAI:
        """
        Instancie le client OpenAI avec un timeout httpx explicite.

        Timeouts choisis :
          - connect : 15s  — délai pour établir la connexion TCP
          - write   : 300s — temps d'upload du fichier audio (gros fichiers)
          - read    : self.timeout_seconds — temps de transcription réel
          - pool    : 15s  — acquisition d'une connexion dans le pool httpx
        """
        timeout = httpx.Timeout(
            connect=15.0,
            write=300.0,
            read=self.timeout_seconds,
            pool=15.0,
        )
        return OpenAI(
            base_url=self.api_base_url,
            api_key="not-needed",      # Requis syntaxiquement mais ignoré par faster-whisper-server
            timeout=timeout,
            max_retries=0,             # Voir module docstring — les retries sont gérés par run.py
        )

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
    ) -> Dict[str, List[Document]]:
        """
        Transcrit chaque source audio/vidéo et retourne les Documents.

        Seules les sources reconnues comme audio ou vidéo sont transcrites ;
        les autres sont ignorées avec un warning.

        Args:
            sources : Liste de chemins (str | Path) ou de ByteStream Haystack.

        Returns:
            {"documents": [Document, ...]}
        """
        if not sources:
            return {"documents": []}

        client = self._build_client()
        documents: List[Document] = []

        for source in sources:
            try:
                doc = self._transcribe_one(client, source)
                if doc is not None:
                    documents.append(doc)
            except Exception as exc:
                # On logue l'erreur et on propage pour que run.py
                # puisse gérer le retry via IngestionPendingFilesError.
                src_label = str(source) if not isinstance(source, ByteStream) else "<ByteStream>"
                logger.error(
                    "whisper.transcription_failed",
                    source=src_label,
                    error=str(exc),
                )
                raise

        return {"documents": documents}

    def _transcribe_one(
        self,
        client: OpenAI,
        source: Union[str, Path, ByteStream],
    ) -> Optional[Document]:
        """
        Transcrit un unique fichier audio/vidéo.

        Returns:
            Document Haystack avec le texte transcrit, ou None si la source
            est ignorée (ex : extension non reconnue).
        """
        # --- Résolution du chemin et validation ---
        if isinstance(source, ByteStream):
            file_path_str = (source.meta or {}).get("file_path", "")
            file_path = Path(file_path_str) if file_path_str else None
        else:
            file_path = Path(source)
            file_path_str = str(file_path.resolve())

        if file_path is None or file_path.suffix.lower() not in _AUDIO_VIDEO_EXTENSIONS:
            logger.warning(
                "whisper.unsupported_source_skipped",
                source=str(source),
            )
            return None

        abs_path_str = str(file_path.resolve())

        logger.info(
            "whisper.transcription_start",
            path=abs_path_str,
            timeout_seconds=self.timeout_seconds,
        )

        # --- Appel API Whisper ---
        with open(abs_path_str, "rb") as audio_file:
            kwargs = dict(
                model=self.model,
                file=(file_path.name, audio_file, _resolve_mime(file_path)),
                response_format="text",
            )
            if self.language:
                kwargs["language"] = self.language

            transcript: str = client.audio.transcriptions.create(**kwargs)

        if not transcript or not transcript.strip():
            logger.warning("whisper.empty_transcript", path=abs_path_str)
            return None

        logger.info(
            "whisper.transcription_done",
            path=abs_path_str,
            chars=len(transcript),
        )

        return Document(
            content=transcript.strip(),
            meta={
                "file_path": abs_path_str,
                "file_name": file_path.name,
                "source_type": "audio_transcript",
            },
        )
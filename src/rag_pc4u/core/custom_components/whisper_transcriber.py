"""
Composant Haystack custom pour la transcription audio via un serveur Whisper distant.

FIXES appliqués :
  1. Timeout httpx explicite (défaut 7200s = 2h) — évite les retries prématurés
     du client OpenAI qui causaient une cascade de requêtes en parallèle.
  2. max_retries=0 — les retries sont délégués à run.py (IngestionPendingFilesError).
  3. Split interne adapté à la parole (audio_chunk_words / audio_chunk_overlap) —
     le transcript brut est découpé en chunks de ~1m30 de parole avant d'être
     envoyé dans le pipeline, avec respect des frontières de phrases.
     → Connecter audio_converter.documents vers joiner_txt.documents dans pipeline.py.
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


# Constantes

_DEFAULT_TIMEOUT_SECONDS = 7_200.0  # 2 heures

# Débit oral moyen du français : ~135 mots/minute.
# 200 mots ≈ 1m30 de parole → chunk sémantiquement cohérent pour réunion/tuto.
# 30 mots d'overlap ≈ 15s de chevauchement → préserve le contexte aux jointures.
_DEFAULT_AUDIO_CHUNK_WORDS = 200
_DEFAULT_AUDIO_CHUNK_OVERLAP = 30

_AUDIO_VIDEO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".flac",
    ".mp4", ".mov", ".mkv", ".avi", ".webm",
}

# Marqueurs de fin de phrase reconnus dans les transcriptions Whisper
_SENTENCE_ENDINGS = {".", "!", "?", "...", "»", '"', "?»", "!»"}



# Helpers


def _resolve_mime(path: Path) -> str:
    """Détermine un MIME type à partir de l'extension, avec fallback."""
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _split_transcript(
    text: str,
    chunk_words: int,
    overlap_words: int,
    base_meta: dict,
) -> List[Document]:
    """
    Découpe un transcript en chunks adaptés à la parole.

    Stratégie :
      - Découpage par mots avec fenêtre glissante.
      - Dans la zone [80 %, 100 %] de chaque fenêtre, on cherche la dernière
        frontière de phrase (., !, ?, ...) pour ne pas couper en plein milieu
        d'une idée. Si aucune frontière n'est trouvée, on coupe à chunk_words.
      - Chevauchement de `overlap_words` mots entre chunks consécutifs pour
        préserver le contexte aux jointures.
      - Si le texte est plus court que chunk_words, un seul Document est produit.

    Args:
        text         : Transcript brut retourné par Whisper.
        chunk_words  : Taille cible d'un chunk en mots.
        overlap_words: Mots partagés entre deux chunks consécutifs.
        base_meta    : Métadonnées communes injectées dans chaque Document produit.

    Returns:
        Liste de Documents Haystack prêts pour le pipeline.
    """
    text = text.strip()
    if not text:
        return []

    words = text.split()

    # Texte court : un seul Document, pas de découpage nécessaire
    if len(words) <= chunk_words:
        return [
            Document(
                content=text,
                meta={**base_meta, "chunk_index": 0, "total_chunks": 1},
            )
        ]

    chunks_text: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk_slice = words[start:end]

        # Ajustement sur frontière de phrase si on n'est pas à la fin du texte
        if end < len(words):
            search_from = max(start + int(chunk_words * 0.8), start + 1)
            last_boundary: Optional[int] = None

            for i in range(end - 1, search_from - 1, -1):
                token = words[i].rstrip()
                if any(token.endswith(marker) for marker in _SENTENCE_ENDINGS):
                    last_boundary = i + 1
                    break

            if last_boundary is not None:
                chunk_slice = words[start:last_boundary]
                end = last_boundary

        chunks_text.append(" ".join(chunk_slice))

        # Prochain chunk : recule de `overlap_words` pour le chevauchement
        next_start = end - overlap_words
        if next_start <= start:
            next_start = start + 1  # sécurité anti-boucle infinie
        start = next_start

    total = len(chunks_text)
    logger.info(
        "whisper.split_done",
        total_words=len(words),
        chunks_produced=total,
        chunk_words=chunk_words,
        overlap_words=overlap_words,
    )

    return [
        Document(
            content=chunk_text,
            meta={**base_meta, "chunk_index": i, "total_chunks": total},
        )
        for i, chunk_text in enumerate(chunks_text)
    ]



# Composant Haystack


@component
class RemoteWhisperTranscriber:
    """
    Composant Haystack qui envoie des fichiers audio/vidéo à un serveur
    Whisper distant (compatible OpenAI /v1/audio/transcriptions), transcrit
    le contenu, puis le découpe en chunks adaptés à la parole avant de
    retourner des Documents Haystack.

    Conçu pour être connecté à joiner_txt.documents dans le pipeline
    (les chunks passent ensuite par cleaner → splitter → joiner_main).
    Comme les chunks audio sont plus petits que le chunk_size du splitter
    (ex : 200 < 548), le splitter les laisse passer tels quels.

    Args:
        api_base_url       : URL de base du serveur Whisper.
                             Ex : "http://rag-whisper:8000/v1"
        model              : Nom du modèle Whisper déclaré côté serveur.
                             Peut être ignoré par le serveur s'il n'expose
                             qu'un seul modèle.
        language           : Code BCP-47 pour forcer la langue (ex : "fr").
                             Si None, Whisper détecte automatiquement.
        timeout_seconds    : Timeout HTTP pour la phase de lecture (= durée max
                             de transcription tolérée). Défaut : 7 200 s (2h).
        audio_chunk_words  : Taille cible d'un chunk en mots.
                             Défaut : 200 mots ≈ 1m30 de parole française.
                             Doit rester inférieur au chunk_size du pipeline.
        audio_chunk_overlap: Mots partagés entre chunks consécutifs.
                             Défaut : 30 mots ≈ 15s de chevauchement.
    """

    def __init__(
        self,
        api_base_url: str,
        model: str = "Systran/faster-whisper-large-v3",
        language: Optional[str] = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        audio_chunk_words: int = _DEFAULT_AUDIO_CHUNK_WORDS,
        audio_chunk_overlap: int = _DEFAULT_AUDIO_CHUNK_OVERLAP,
    ):
        self.api_base_url = api_base_url
        self.model = model
        self.language = language
        self.timeout_seconds = timeout_seconds
        self.audio_chunk_words = audio_chunk_words
        self.audio_chunk_overlap = audio_chunk_overlap

    def _build_client(self) -> OpenAI:
        """
        Instancie le client OpenAI avec un timeout httpx explicite.

        Timeouts :
          connect : 15s  — établissement de la connexion TCP
          write   : 300s — upload du fichier audio (gros fichiers)
          read    : self.timeout_seconds — durée de transcription
          pool    : 15s  — acquisition d'une connexion dans le pool httpx
        """
        timeout = httpx.Timeout(
            connect=15.0,
            write=300.0,
            read=self.timeout_seconds,
            pool=15.0,
        )
        return OpenAI(
            base_url=self.api_base_url,
            api_key="not-needed",
            timeout=timeout,
            max_retries=0,  # retries gérés par run.py via IngestionPendingFilesError
        )

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
    ) -> Dict[str, List[Document]]:
        """
        Transcrit chaque source audio/vidéo, découpe le transcript en chunks
        adaptés à la parole et retourne la liste de Documents résultants.

        Seules les extensions reconnues dans _AUDIO_VIDEO_EXTENSIONS sont
        traitées. Les autres sont ignorées avec un warning.

        Args:
            sources : Chemins (str | Path) ou ByteStream Haystack.

        Returns:
            {"documents": [Document, ...]}
        """
        if not sources:
            return {"documents": []}

        client = self._build_client()
        documents: List[Document] = []

        for source in sources:
            try:
                chunks = self._transcribe_and_split(client, source)
                documents.extend(chunks)
            except Exception as exc:
                src_label = (
                    str(source)
                    if not isinstance(source, ByteStream)
                    else "<ByteStream>"
                )
                logger.error(
                    "whisper.transcription_failed",
                    source=src_label,
                    error=str(exc),
                )
                raise  # propagé vers run.py → IngestionPendingFilesError

        return {"documents": documents}

    def _transcribe_and_split(
        self,
        client: OpenAI,
        source: Union[str, Path, ByteStream],
    ) -> List[Document]:
        """
        Transcrit un unique fichier et retourne ses chunks.

        Returns:
            Liste de Documents (vide si source ignorée ou transcript vide).
        """
        # Résolution du chemin
        if isinstance(source, ByteStream):
            file_path_str = (source.meta or {}).get("file_path", "")
            file_path = Path(file_path_str) if file_path_str else None
        else:
            file_path = Path(source)

        if file_path is None or file_path.suffix.lower() not in _AUDIO_VIDEO_EXTENSIONS:
            logger.warning("whisper.unsupported_source_skipped", source=str(source))
            return []

        abs_path_str = str(file_path.resolve())

        logger.info(
            "whisper.transcription_start",
            path=abs_path_str,
            timeout_seconds=self.timeout_seconds,
            audio_chunk_words=self.audio_chunk_words,
        )

        # Appel API Whisper
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
            return []

        logger.info(
            "whisper.transcription_done",
            path=abs_path_str,
            chars=len(transcript),
            words=len(transcript.split()),
        )

        # Découpage en chunks adaptés à la parole
        base_meta = {
            "file_path": abs_path_str,
            "file_name": file_path.name,
            "source_type": "audio_transcript",
        }

        return _split_transcript(
            text=transcript,
            chunk_words=self.audio_chunk_words,
            overlap_words=self.audio_chunk_overlap,
            base_meta=base_meta,
        )
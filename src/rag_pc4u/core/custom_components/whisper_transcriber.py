"""
Composant Haystack custom pour la transcription audio via un serveur Whisper distant.

FIXES appliqués :
  1. Timeout httpx explicite (défaut 86400s = 1j) — évite les retries prématurés
     du client OpenAI qui causaient une cascade de requêtes en parallèle.
  2. max_retries=0 — les retries sont délégués à run.py (IngestionPendingFilesError).
  3. Split basé sur les segments Whisper (verbose_json) plutôt que sur le texte brut —
     chaque chunk regroupe des segments consécutifs jusqu'à audio_chunk_words mots,
     avec un overlap de audio_chunk_overlap mots sur le chunk suivant.
     Les timestamps start/end de chaque chunk sont conservés dans les métadonnées.
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

_DEFAULT_TIMEOUT_SECONDS = 86_400.0  # 1jours

# Débit oral moyen du français : ~135 mots/minute.
# 200 mots ≈ 1m30 de parole → chunk sémantiquement cohérent pour réunion/tuto.
# 30 mots d'overlap ≈ 15s de chevauchement → préserve le contexte aux jointures.
_DEFAULT_AUDIO_CHUNK_WORDS = 200
_DEFAULT_AUDIO_CHUNK_OVERLAP = 30

_AUDIO_VIDEO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".flac",
    ".mp4", ".mov", ".mkv", ".avi", ".webm",
}





# Helpers


def _resolve_mime(path: Path) -> str:
    """Détermine un MIME type à partir de l'extension, avec fallback."""
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _split_segments(
    segments: List[dict],
    chunk_words: int,
    overlap_words: int,
    base_meta: dict,
) -> List[Document]:
    """
    Regroupe les segments Whisper (verbose_json) en chunks adaptés à la parole.

    Stratégie :
      - On accumule des segments consécutifs jusqu'à atteindre chunk_words mots.
        Chaque segment est déjà une unité sémantique naturelle produite par Whisper
        (frontière de phrase respectée), donc on ne coupe jamais en plein milieu
        d'une idée — contrairement à un split naïf par mots sur le texte brut.
      - Les timestamps start/end du premier et dernier segment du chunk sont
        conservés dans les métadonnées → permet de retrouver la minute exacte
        dans l'audio source lors d'une recherche RAG.
      - Overlap : les `overlap_words` derniers mots du chunk N sont réinjectés
        en tête du chunk N+1 sous forme de texte (pas de segments entiers),
        pour préserver le contexte aux jointures sans dupliquer les timestamps.
      - Si le transcript complet est plus court que chunk_words, un seul
        Document est produit.

    Args:
        segments     : Liste de segments Whisper (champs attendus : "text",
                       "start", "end"). Retournés par verbose_json.
        chunk_words  : Taille cible d'un chunk en mots.
                       Débit oral français ~135 mots/min → 200 mots ≈ 1m30.
        overlap_words: Mots de chevauchement entre chunks consécutifs.
                       30 mots ≈ 15s de chevauchement.
        base_meta    : Métadonnées communes injectées dans chaque Document.

    Returns:
        Liste de Documents Haystack prêts pour le pipeline.
    """
    if not segments:
        return []

    documents: List[Document] = []
    current_segments: List[dict] = []
    current_word_count = 0
    overlap_prefix = ""  # texte d'overlap issu du chunk précédent

    def _flush(segs: List[dict], prefix: str, chunk_idx: int) -> str:
        """Construit un Document à partir des segments accumulés, retourne le texte d'overlap."""
        text = (prefix + " " + " ".join(s["text"].strip() for s in segs)).strip()
        documents.append(
            Document(
                content=text,
                meta={
                    **base_meta,
                    "chunk_index": chunk_idx,
                    # Timestamps : début du 1er segment, fin du dernier
                    "start_time": segs[0]["start"],
                    "end_time": segs[-1]["end"],
                },
            )
        )
        # Calcul de l'overlap : on prend les overlap_words derniers mots du texte produit
        words = text.split()
        return " ".join(words[-overlap_words:]) if len(words) > overlap_words else ""

    for seg in segments:
        seg_words = len(seg["text"].split())
        current_segments.append(seg)
        current_word_count += seg_words

        if current_word_count >= chunk_words:
            overlap_prefix = _flush(current_segments, overlap_prefix, len(documents))
            current_segments = []
            current_word_count = 0

    # Flush du dernier chunk (peut être plus court que chunk_words)
    if current_segments:
        _flush(current_segments, overlap_prefix, len(documents))

    total = len(documents)
    logger.info(
        "whisper.split_done",
        total_segments=len(segments),
        chunks_produced=total,
        chunk_words=chunk_words,
        overlap_words=overlap_words,
    )

    # Injection du total dans les métadonnées maintenant qu'on le connaît
    for doc in documents:
        doc.meta["total_chunks"] = total

    return documents



# Composant Haystack


@component
class RemoteWhisperTranscriber:
    """
    Composant Haystack qui envoie des fichiers audio/vidéo à un serveur
    Whisper distant (compatible OpenAI /v1/audio/transcriptions), transcrit
    le contenu, puis le découpe en chunks basés sur les segments Whisper
    avant de retourner des Documents Haystack.

    Le chunking utilise verbose_json pour récupérer les segments natifs de
    Whisper (frontières de phrases déjà respectées) plutôt que de découper
    le texte brut par mots. Chaque chunk conserve les timestamps start/end
    dans ses métadonnées, ce qui permet de retrouver la minute exacte dans
    l'audio source lors d'une recherche RAG.

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
                             de transcription tolérée). Défaut : 86 400 s (1j).
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

        # Appel API Whisper — verbose_json pour récupérer les segments natifs
        # avec leurs timestamps (start/end par phrase).
        with open(abs_path_str, "rb") as audio_file:
            kwargs = dict(
                model=self.model,
                file=(file_path.name, audio_file, _resolve_mime(file_path)),
                response_format="verbose_json",
            )
            if self.language:
                kwargs["language"] = self.language

            response = client.audio.transcriptions.create(**kwargs)

        segments = response.segments or []
        full_text = response.text or ""

        if not full_text.strip():
            logger.warning("whisper.empty_transcript", path=abs_path_str)
            return []

        logger.info(
            "whisper.transcription_done",
            path=abs_path_str,
            chars=len(full_text),
            words=len(full_text.split()),
            segments=len(segments),
        )

        # Découpage en chunks basés sur les segments Whisper
        base_meta = {
            "file_path": abs_path_str,
            "file_name": file_path.name,
            "source_type": "audio_transcript",
        }

        return _split_segments(
            segments=segments,
            chunk_words=self.audio_chunk_words,
            overlap_words=self.audio_chunk_overlap,
            base_meta=base_meta,
        )
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# ── Dépendances système ───────────────────────────────────────────────────────
#
# Groupe 1 — OpenCV / libGL (requis par Docling pour le modèle de layout)
#   libgl1        → libGL.so.1 — erreur "cannot open shared object file" sans ça
#   libglib2.0-0  → libgthread-2.0.so.0 — requis avec libGL sur slim
#   libsm6        → Session Management — requis par OpenCV headless sur slim
#   libxext6      → X11 extensions — requis par OpenCV sur slim
#   libxrender1   → X Rendering — requis par OpenCV sur slim
#
# Groupe 2 — traitement PDF / images
#   poppler-utils → pdfinfo, pdftoppm — rendu de pages PDF par docling-parse
#                   (tire libpoppler-cpp automatiquement en dépendance)
#   libfreetype6  → rendu de polices dans les PDFs
#
# Groupe 3 — parallélisme / calcul
#   libgomp1      → OpenMP — requis par PyTorch CPU et FastEmbed
#
# Groupe 4 — OCR (déjà présents dans l'original, conservés)
#   tesseract-ocr, tesseract-ocr-fra, ghostscript, qpdf, unpaper
#
# Groupe 5 — réseau / certificats
#   curl, ca-certificates
#
# NOTE : les paquets versionnés (libpoppler-cpp0v5, libjpeg62-turbo, etc.)
# ont été retirés car leurs noms changent entre Debian 12 (bookworm) et
# Debian 13 (trixie). Les dépendances transitives de poppler-utils et
# tesseract-ocr les tirent automatiquement dans la bonne version.

RUN apt-get update && apt-get install -y --no-install-recommends \
        \
        # OpenCV / libGL  OBLIGATOIRE pour Docling layout model \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        \
        # Traitement PDF / images
        poppler-utils \
        libfreetype6 \
        \
        # Parallélisme (PyTorch CPU, FastEmbed)
        libgomp1 \
        \
        # OCR
        tesseract-ocr \
        tesseract-ocr-fra \
        ghostscript \
        qpdf \
        unpaper \
        \
        # Réseau
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-dev

EXPOSE 8000
EXPOSE 8001

CMD ["uv", "run", "rag-api"]
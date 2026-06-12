"""Le script se lance UNE SEULE FOIS pour cacher tous les modèles localement."""
import os
import subprocess
import sys
from pathlib import Path

# 1. Configuration des chemins (AVANT tout import ML)
LOCAL_MODELS_DIR = Path(__file__).parent / "models_cache"

HF_CACHE      = LOCAL_MODELS_DIR / "hf_cache"
FASTEMBED_DIR = LOCAL_MODELS_DIR / "fastembed_cache"
DOCLING_DIR   = LOCAL_MODELS_DIR / "docling_cache"

for d in (HF_CACHE, FASTEMBED_DIR, DOCLING_DIR):
    d.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"]                = str(HF_CACHE)
os.environ["FASTEMBED_CACHE_PATH"]   = str(FASTEMBED_DIR)
os.environ["DOCLING_ARTIFACTS_PATH"] = str(DOCLING_DIR)

# 2. Imports après configuration de l'environnement
from fastembed import SparseTextEmbedding
from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder

print("1/4 — Caching BM25 (FastEmbed)...")
SparseTextEmbedding(model_name="Qdrant/bm25")

print("2/4 — Caching BAAI/bge-m3 tokenizer (Docling HybridChunker)...")
snapshot_download(
    repo_id="BAAI/bge-m3",
    ignore_patterns=["*.safetensors", "*.bin"],
)

print("3/4 — Caching BAAI/bge-reranker-v2-m3 (CrossEncoder)...")
CrossEncoder("BAAI/bge-reranker-v2-m3")

print("4/4 — Caching modèles Docling (layout + tableformer) via docling-tools...")
result = subprocess.run(
    ["docling-tools", "models", "download", "--output-dir", str(DOCLING_DIR)],
    env={**os.environ},
)

if result.returncode != 0:
    print("\n✗ docling-tools models download a échoué.")
    print("  Lance manuellement :")
    print(f"  DOCLING_ARTIFACTS_PATH={DOCLING_DIR} uv run docling-tools models download --output-dir {DOCLING_DIR}")
    sys.exit(1)

# Vérification : les .safetensors doivent être présents
safetensors_files = list(DOCLING_DIR.rglob("*.safetensors"))
if not safetensors_files:
    print("\n✗ Aucun fichier .safetensors trouvé dans docling_cache/.")
    print("  Le téléchargement a peut-être échoué silencieusement.")
    sys.exit(1)

print("   Modèles Docling présents :")
for f in safetensors_files:
    print(f"   ✓ {f.relative_to(DOCLING_DIR)}")

print("\n✅ Tous les modèles sont cachés.")
print(f"   HF cache       : {HF_CACHE}")
print(f"   FastEmbed cache: {FASTEMBED_DIR}")
print(f"   Docling cache  : {DOCLING_DIR}")

print()
for label, path in [
    ("hf_cache",        HF_CACHE),
    ("fastembed_cache", FASTEMBED_DIR),
    ("docling_cache",   DOCLING_DIR),
]:
    items = list(path.rglob("*")) if path.exists() else []
    files = [x for x in items if x.is_file()]
    status = "✓" if files else "✗ VIDE — quelque chose a raté"
    print(f"   {status}  {label}/ ({len(files)} fichiers)")

print("\nHF_HUB_OFFLINE=1 peut maintenant être activé.")
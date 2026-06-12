# RAG PC4U

Solution RAG on-premise pour PC4U. Les documents sont indexés dans Qdrant via Haystack, les embeddings et le LLM sont servis par Ollama, et la synchronisation des sources se fait automatiquement depuis Nextcloud.

---

## Ce que ça expose

| Port | Service |
|------|---------|
| `8000` | API RAG — interface compatible OpenAI (Open WebUI, curl, etc.) |
| `8001` | Dashboard de synchronisation Nextcloud → Qdrant |

---

## Infrastructure attendue

Ces trois composants doivent être accessibles depuis le LXC avant de démarrer quoi que ce soit.

| Composant | Adresse par défaut |
|-----------|-------------------|
| Qdrant    | `192.168.204.20:6333` |
| Ollama    | `192.168.204.21:11434` |
| Nextcloud | à définir dans `.env` |

Sur le serveur Ollama, les deux modèles doivent être pullés :

```bash
ollama pull bge-m3:latest
ollama pull qwen2.5:14b
```

---

## Mise en place depuis le LXC

### 1. Docker

Si Docker n'est pas encore installé sur le LXC :

```bash
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker
```

```bash
# Vérification
docker compose version
```

### 2. uv

`uv` est nécessaire pour pré-cacher les modèles HuggingFace sur l'hôte avant le premier démarrage (voir étape 5).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

```bash
# Vérification
uv --version
```

### 3. Récupérer le projet

```bash
mkdir ramcn
cd ramcn
git clone https://github.com/flags32/rag-pc4u.git 
cd /root/ramcn/rag-pc4u
```

> **Chemin à ne pas changer.** Le `docker-compose.yml` monte les caches de modèles avec des chemins absolus qui pointent vers `/root/ramcn/rag-pc4u/`. Si vous clonez ailleurs, mettez à jour les deux entrées `volumes` dans `docker/docker-compose.yml`.

### 4. Configurer l'environnement

```bash
cp .env.example .env
nano .env
```

Variables à renseigner dans `.env` :

```env
# Nextcloud — WebDAV
NEXTCLOUD_URL=http://192.168.x.x
NEXTCLOUD_USER=mon_utilisateur
NEXTCLOUD_PASSWORD=mon_mot_de_passe

```

Les variables Ollama et Qdrant sont déjà définies dans `docker-compose.yml` et n'ont pas à être dupliquées dans `.env`. Si vous les ajoutez quand même, celles du `docker-compose.yml` auront la priorité (elles écrasent `env_file`).

### 5. Pré-cacher les modèles — étape obligatoire

Les containers tournent en mode hors-ligne (`HF_HUB_OFFLINE=1`). Le téléchargement des modèles doit se faire **une fois** depuis le LXC, avant le premier démarrage.

```bash
cd /root/ramcn/rag-pc4u
uv sync --frozen
uv run python src/rag_pc4u/scripts/precache_models.py
```
le code vous le dira mais pensais bien après avoir executé le script de cache_models.py a faire :
```.env
HF_HUB_OFFLINE=1
```

Ce script télécharge et met en cache trois choses :

- **`Qdrant/bm25`** — modèle sparse BM25 via FastEmbed
- **`BAAI/bge-m3`** — tokenizer uniquement (utilisé par Docling pour le chunking, pas les poids du modèle complet)
- **`BAAI/bge-reranker-v2-m3`** — poids complets du reranker (~1,1 Go)

Durée estimée : 10 à 20 minutes selon votre connexion. À ne faire qu'une seule fois — ou si vous changez de machine.

Vérification une fois terminé :

```bash
ls src/rag_pc4u/scripts/models_cache/hf_cache/
ls src/rag_pc4u/scripts/models_cache/fastembed_cache/
```

Les deux dossiers doivent être non vides.

### 6. Builder et démarrer

```bash
cd /root/ramcn/rag-pc4u/docker
docker compose up --build -d
```

Le build installe toutes les dépendances Python via `uv` dans l'image. PyTorch est tiré depuis l'index CPU (`download.pytorch.org/whl/cpu`) — pas de GPU requis côté container. Comptez 10 à 15 minutes pour le premier build.

```bash
# Suivre les logs
docker compose logs -f

# Un service en particulier
docker compose logs -f api
docker compose logs -f dashboard
```

L'API est prête quand vous voyez :

```
rag-api | INFO:     Application startup complete.
```

---

## Utilisation

### API RAG (port 8000)

L'API est compatible avec le format OpenAI. Dans **Open WebUI**, créez une connexion de type « OpenAI » avec :

- **URL** : `http://<IP_DU_LXC>:8000`
- **Clé API** : n'importe quelle valeur (non vérifiée)

Les collections Qdrant s'affichent comme des modèles dans le sélecteur. Chaque collection est un espace documentaire indépendant — pas de mélange entre les sources.

Test rapide depuis le LXC :

```bash
# Liste des collections disponibles
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Question RAG sur une collection
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "documents_default",
    "messages": [{"role": "user", "content": "Qu'\''est-ce qu'\''un variateur de vitesse ?"}]
  }' | python3 -m json.tool
```

### Dashboard Nextcloud (port 8001)

Accessible à `http://<IP_DU_LXC>:8001`.

Depuis le dashboard, vous pouvez :

- Créer un mapping `dossier Nextcloud → collection Qdrant` avec un intervalle de synchronisation
- Déclencher une synchronisation manuelle à tout moment
- Consulter l'historique des syncs et les éventuelles erreurs

La première synchronisation d'un nouveau mapping démarre immédiatement en arrière-plan et indexe l'intégralité du dossier distant. Les syncs suivantes sont incrémentales : seuls les fichiers nouveaux, modifiés ou supprimés sont traités.

Formats supportés : `.txt`, `.md`, `.pdf`, `.csv` et fichiers sans extension.

---

## Commandes utiles

```bash
# Arrêter tous les services
docker compose down

# Redémarrer un service sans rebuild
docker compose restart api
docker compose restart dashboard

# Rebuild d'un service après modification du code
docker compose up --build -d api

# Voir les dernières lignes de log
docker compose logs --tail=100 api

# Ouvrir un shell dans un container
docker exec -it rag-api bash

# Vérifier que les caches sont bien montés dans le container
docker exec rag-api ls /root/.cache/hf_cache/
docker exec rag-api ls /root/.cache/fastembed_cache/

# Vérifier la connexion à Qdrant depuis le container
docker exec rag-api curl -s http://192.168.204.20:6333/collections | python3 -m json.tool

# Vérifier la connexion à Ollama depuis le container
docker exec rag-api curl -s http://192.168.204.21:11434/api/tags | python3 -m json.tool
```

---

## Dépannage

### "couldn't find files in cached files" au démarrage de l'API

Le container ne trouve pas les modèles dans son cache. Causes possibles :

```bash
# 1. Vérifier que les fichiers existent sur l'hôte
ls /root/ramcn/rag-pc4u/src/rag_pc4u/scripts/models_cache/hf_cache/

# 2. Vérifier que les volumes sont bien montés
docker exec rag-api ls /root/.cache/hf_cache/
```

Si le dossier est vide sur l'hôte, le script de cache n'a pas été lancé ou a échoué — relancez l'étape 5.

Si le dossier est plein sur l'hôte mais vide dans le container, le volume n'est pas monté correctement — vérifiez les chemins dans `docker/docker-compose.yml` et que le projet est bien à `/root/ramcn/rag-pc4u/`.

### Le dashboard affiche "Nextcloud non joignable"

Tester la connexion WebDAV directement depuis le LXC :

```bash
curl -u UTILISATEUR:MOT_DE_PASSE \
  http://NEXTCLOUD_URL/remote.php/dav/files/UTILISATEUR/ \
  -X PROPFIND \
  -H "Depth: 0"
```

Une réponse 207 confirme que les credentials et l'URL sont corrects.

### Un port est déjà utilisé

```bash
ss -tlnp | grep -E '8000|8001'
```

Identifier le processus concerné et l'arrêter, ou changer les ports côté `docker-compose.yml` (partie gauche des mappings `host:container`).

### Mettre à jour le projet

```bash
cd /root/ramcn/rag-pc4u
git pull
cd docker
docker compose up --build -d
```

Si les dépendances Python ont changé (`pyproject.toml` ou `uv.lock` modifiés), le rebuild le détecte automatiquement et réinstalle ce qui a changé.

---

## Structure du projet

```
rag-pc4u/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/
│   └── rag_pc4u/
│       ├── api/              # Point d'entrée FastAPI principal (port 8000)
│       ├── core/             # Config, composants partagés, logger
│       ├── ingestion/        # Pipeline d'indexation + watcher Nextcloud
│       ├── retrieval/        # Pipeline RAG hybride + reranker + prompts
│       ├── dashboard/        # Dashboard FastAPI (port 8001) + scheduler
│       └── scripts/
│           ├── cache_models.py          # À lancer une fois avant Docker
│           └── models_cache/            # Monté en volume dans les containers
│               ├── hf_cache/
│               └── fastembed_cache/
├── pyproject.toml
├── uv.lock
└── README.md                # Ce fichier — requis par le Dockerfile
```

> `README.md` est copié dans l'image Docker lors du build (`COPY pyproject.toml uv.lock README.md ./`). Il doit donc être présent à la racine du projet.

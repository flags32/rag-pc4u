# Intégration Open WebUI

Open WebUI est l'interface de chat que vous utilisez pour interroger vos collections RAG.
Il communique avec l'API PC4U (port 8000) exactement comme s'il parlait à l'API OpenAI — aucun plugin ou adaptation spécifique n'est nécessaire.

---

## Connexion initiale

Dans Open WebUI, créez une connexion de type **OpenAI** :

**Paramètres > Connexions > OpenAI > Ajouter**

| Champ | Valeur |
|-------|--------|
| URL de base | `http://<IP_DU_LXC>:8000` |
| Clé API | n'importe quelle valeur (ex : `pc4u`) |

> La clé API n'est pas vérifiée côté serveur. N'importe quelle chaîne non vide suffit.

Après avoir sauvegardé, Open WebUI contacte automatiquement `/v1/models` pour récupérer la liste des modèles disponibles.

---

## Collections = Modèles

Chaque collection Qdrant que vous avez configurée apparaît comme un **modèle** dans le sélecteur de Open WebUI.

```
Sélecteur de modèle Open WebUI
  ├── documents_technique
  ├── documents_qualite
  └── manuels_machines
```

Le nom affiché est exactement le `collection_name` défini dans vos mappings Nextcloud.
Changer de collection = changer de base documentaire — il n'y a pas de mélange entre les sources.

Pour ajouter une collection au sélecteur, créez un nouveau mapping depuis le dashboard (port 8001) puis redémarrez l'API (ou attendez la prochaine requête `/v1/models`).

---

## Flux d'une question

Quand vous envoyez un message dans Open WebUI, voici ce qui se passe :

```
Open WebUI
    │
    │  POST /v1/chat/completions
    │  { "model": "documents_technique", "messages": [...] }
    ▼
API RAG (port 8000)
    │
    ├── Détection mot-clé ? (dashboard / synchro / nextcloud …)
    │       └── OUI → retourne lien vers le dashboard, pas de RAG
    │
    └── NON → Pipeline RAG Haystack
                │
                ├── Embedding de la question (bge-m3 via Ollama)
                ├── Recherche hybride dans Qdrant
                │     ├── Dense  (bge-m3)
                │     └── Sparse (BM25 FastEmbed)
                ├── Reranking (bge-reranker-v2-m3, local)
                ├── Génération de la réponse (qwen2.5:14b via Ollama)
                └── Réponse au format OpenAI → Open WebUI l'affiche
```

La réponse retournée est au format `chat.completion` standard — Open WebUI n'a pas besoin de traitement particulier.

---

## Raccourci dashboard depuis le chat

Certains mots-clés dans votre question déclenchent une réponse spéciale au lieu d'interroger Qdrant :

> **Mots-clés détectés** : `dashboard`, `synchro`, `nextcloud`, `synchronisation`, `panneau de contrôle`

Dans ce cas, l'API retourne directement un lien cliquable vers `http://<IP_DU_LXC>:8001` sans passer par le pipeline RAG. Pratique pour accéder au dashboard sans quitter Open WebUI.

Exemple :
```
Vous    : Comment accéder au dashboard de synchronisation ?
RAG     : Voici le lien vers votre panneau de gestion…  🔗 [Ouvrir le dashboard]
```

---

## Streaming

Le paramètre `stream` est accepté dans les requêtes mais la réponse est toujours retournée en un seul bloc (`stream: false`). Open WebUI affiche la réponse complète une fois le pipeline terminé — pas de défilement token par token.

---

## Tester la connexion sans Open WebUI

```bash
# Vérifier que les collections sont bien exposées comme modèles
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Poser une question RAG sur une collection
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "documents_technique",
    "messages": [{"role": "user", "content": "Qu'\''est-ce qu'\''un variateur de vitesse ?"}]
  }' | python3 -m json.tool

# Tester le raccourci dashboard
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "documents_technique",
    "messages": [{"role": "user", "content": "accéder au dashboard nextcloud"}]
  }' | python3 -m json.tool
```

---

## Ce que Open WebUI ne gère pas (côté RAG)

| Fonctionnalité | Comportement |
|----------------|-------------|
| Historique de conversation | Seul le dernier message `role: user` est utilisé comme question. Le contexte des messages précédents est ignoré par le pipeline RAG. |
| Streaming token par token | Non supporté — réponse retournée en bloc. |
| Upload de fichiers depuis Open WebUI | Non supporté — l'indexation passe uniquement par Nextcloud → Dashboard. |
| Sélection multi-collections | Non supporté — une collection par conversation. |
"""Prompts module Rag PC4U pour retrieval"""

RAG_SYSTEM_PROMPT = "Tu es un assistant technique expert. Réponds à la question de l'utilisateur en utilisant uniquement les informations contenues dans les documents fournis. Si tu ne trouves pas la réponse, dis-le simplement."

RAG_USER_TEMPLATE = """
Documents fournis :
{% for doc in documents %}
--- Document {{ loop.index }} ---
{{ doc.content }}
{% endfor %}

Question : {{ query }}
et reponds uniqument en francais 
"""
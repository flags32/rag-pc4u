from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Ollama
    ollama_host: str = "http://192.168.204.21:11434" #modifier en fonction
    ollama_embed_model: str = "bge-m3"
    ollama_llm_model: str = "mistral-small:22b"

    # Qdrant pointe vers le LXC Proxmox
    qdrant_host: str = "http://192.168.204.20"
    qdrant_port: int = 6333
    embedding_dim: int = 1024  # dimension de bge-m3, utilisée à la création de la collection

    # RAG
    top_k: int = 5  # nombre de chunks remontés par le retriever
    chunk_size: int = 512  # taille des chunks pour DocumentSplitter
    chunk_overlap: int = 50  # recouvrement entre chunks

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Cloisonnement client
    client_id: str = "client_demo"
    #version modifiable par client à tester  collection_name: str = f"documents_{client_id}"
    collection_name: str = "documents_client_demo"# à modifier en fonction du client pour l'instant ce n'est pas configurable et c'est de la demo
    # SMB
    smb_host: str = ""
    smb_share: str = ""
    smb_user: str = ""
    smb_password: str = ""

    # Nextcloud / WebDAV
    nextcloud_url: str = ""
    nextcloud_user: str = ""
    nextcloud_password: str = ""
    nextcloud_remote_path: str = "/documents"
    # Logs
    log_level: str = "INFO"

    @property#proprety est utile pour tout ce qui est url ou autre en python car on peut l'appeler directement sans faire un get'
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Ollama
    ollama_host: str = "http://192.168.204.21:11434"
    ollama_embed_model: str = "bge-m3:latest"
    ollama_llm_model: str = "qwen2.5:14b-instruct-q8_0"

    # Qdrant — pointe vers le LXC Proxmox
    qdrant_host: str = "192.168.204.20"
    qdrant_port: int = 6333
    embedding_dim: int = 1024  # dimension de bge-m3

    # RAG
    top_k: int = 35      # chunks remontés par le retriever
    chunk_size: int = 384
    chunk_overlap: int = 40

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    List_collection: list[str] = [] #on va faire une commande pour recuperer la liste des collections qdrant
    default_collection: str = "documents_machine"

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

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


settings = Settings()
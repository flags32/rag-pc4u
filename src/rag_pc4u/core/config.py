from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_embed_model: str = "bge-m3"
    ollama_llm_model: str = "qwen2.5:7b"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Cloisonnement client
    client_id: str = "client_demo"
    collection_name: str = "documents_client_demo"

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

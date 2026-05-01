from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    data_dir: Path = Path("./data")
    top_k: int = 5
    # all-MiniLM-L6-v2 has max_seq_length=256 tokens (~190 English words).
    # Beyond that, the embedder silently truncates — keep chunks under 200 words.
    chunk_tokens: int = 180
    chunk_overlap: int = 40
    history_turns: int = 4

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"


settings = Settings()
settings.index_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)

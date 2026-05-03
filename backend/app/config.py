from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    # Asymmetric embedder trained on (query, passage) pairs — better fit for
    # short user questions retrieving longer document chunks than the
    # symmetric all-MiniLM-L6-v2. Same 384-dim, max_seq_length=512.
    embed_model: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    data_dir: Path = Path("./data")
    top_k: int = 10
    # multi-qa-MiniLM-L6-cos-v1 has max_seq_length=512 tokens (~390 English words).
    # 180 words (~234 tokens) leaves comfortable headroom; if you swap embedders,
    # check the new max_seq_length and adjust.
    chunk_tokens: int = 180
    chunk_overlap: int = 40
    history_turns: int = 4
    # v2 multi-index pipeline: numeric ANN is gated. At a few-thousand-row scale
    # a pandas filter via query_table is faster and exact than FAISS L2.
    enable_numeric_ann: bool = False

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"


settings = Settings()
settings.index_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)

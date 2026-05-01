from collections import defaultdict, deque
from typing import Deque

import pandas as pd

from app.rag.vector_store import VectorStore
from app.config import settings


vector_store = VectorStore(settings.index_dir)
# load_or_init() is called in the FastAPI lifespan (main.py) so startup errors surface clearly.

dataframes_by_file_id: dict[str, pd.DataFrame] = {}
filename_to_file_id: dict[str, str] = {}
documents: dict[str, dict] = {}  # filename -> {file_id, type, chunks}

chat_history: dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=settings.history_turns * 2))

from typing import Literal, NotRequired, TypedDict


SCHEMA_VERSION = 2

ChunkType = Literal[
    "text",
    "table",
    "synopsis",
    "enumeration",
    "tabular_summary",
    "row",
    "entity",
    "numeric_vector",
    "time_window",
]


class ChunkMeta(TypedDict):
    chunk_id: str
    file: str
    sheet: str | None
    row_number: int | None
    chunk_type: ChunkType
    canonical_id: str | None
    ingestion_time: str
    version: int

    text: NotRequired[str]
    page: NotRequired[int | None]
    aliases: NotRequired[list[str]]
    column_names: NotRequired[list[str]]
    numeric_columns: NotRequired[list[str]]
    window: NotRequired[Literal["30d", "90d", "365d"]]
    stats: NotRequired[dict]
    source_chunk_ids: NotRequired[list[str]]
    file_id: NotRequired[str]

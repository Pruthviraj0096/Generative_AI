import json
import os
import re
from typing import Any
from urllib.parse import quote_plus

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text


load_dotenv()

DB_USER = os.getenv("DB_USER", "devread_write")
DB_PASSWORD = os.getenv("DB_PASSWORD", "devread_write@123")
DB_HOST = os.getenv("DB_HOST", "devdb.mdindia.com")
DB_PORT = os.getenv("DB_PORT", "5000")
DB_NAME = os.getenv("DB_NAME", "apachedbdev")
DB_SCHEMA = os.getenv("DB_SCHEMA", "icd")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://m9960-bk.mdindia.com:11434")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def get_connection_string(db_name: str | None = None) -> str:
    user = os.getenv("DB_USER", DB_USER)
    password = quote_plus(os.getenv("DB_PASSWORD", DB_PASSWORD))
    host = os.getenv("DB_HOST", DB_HOST)
    port = os.getenv("DB_PORT", DB_PORT)
    database = db_name or os.getenv("DB_NAME", DB_NAME)
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def get_embeddings_model(source: str, model_name: str, api_key: str | None, ollama_base_url: str | None):
    if source.lower() == "ollama":
        return OllamaEmbeddings(model=model_name, base_url=ollama_base_url)
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key is required for OpenAI embeddings.")
    return OpenAIEmbeddings(openai_api_key=api_key, model=model_name)


def parse_vector_string(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.array(value, dtype=float)

    value = str(value).strip()
    if not value:
        return None

    try:
        return np.array(json.loads(value), dtype=float)
    except Exception:
        numbers = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", value)
        if numbers:
            return np.array([float(number) for number in numbers], dtype=float)
    return None


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float | None:
    if vector_a.shape != vector_b.shape:
        return None

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return None

    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def validate_identifier(value: str, field_name: str) -> str:
    if not IDENTIFIER_PATTERN.match(value):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must contain only letters, numbers, and underscores, and cannot start with a number.",
        )
    return value


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Text to search against stored embeddings.")
    database: str = DB_NAME
    schema_name: str = DB_SCHEMA
    table_name: str = "documents"
    text_column: str = "content"
    vector_column: str = "embedding"
    limit: int = Field(default=5, ge=1, le=50)
    embedding_source: str = "Ollama"
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str | None = DEFAULT_OLLAMA_BASE_URL
    openai_api_key: str | None = None


class SearchResult(BaseModel):
    rank: int
    similarity: float
    distance: float
    row: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    searched_rows: int
    valid_vectors: int
    results: list[SearchResult]


app = FastAPI(title="Embedding Search API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


def run_embedding_search(request: SearchRequest) -> SearchResponse:
    schema_name = validate_identifier(request.schema_name, "schema_name")
    table_name = validate_identifier(request.table_name, "table_name")
    text_column = validate_identifier(request.text_column, "text_column")
    vector_column = validate_identifier(request.vector_column, "vector_column")

    model = get_embeddings_model(
        request.embedding_source,
        request.embedding_model,
        request.openai_api_key or os.getenv("OPENAI_API_KEY"),
        request.ollama_base_url,
    )
    query_vector = np.array(model.embed_query(request.query), dtype=float)

    query = text(
        f'SELECT * FROM "{schema_name}"."{table_name}" '
        f'WHERE "{text_column}" IS NOT NULL '
        f'AND "{vector_column}" IS NOT NULL'
    )

    try:
        engine = create_engine(get_connection_string(request.database))
        with engine.connect() as conn:
            result_proxy = conn.execute(query)
            rows = result_proxy.fetchall()
            column_names = list(result_proxy.keys())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

    matches = []
    valid_vectors = 0

    for row in rows:
        row_dict = dict(zip(column_names, row))
        db_vector = parse_vector_string(row_dict.get(vector_column))
        if db_vector is None:
            continue

        similarity = cosine_similarity(query_vector, db_vector)
        if similarity is None:
            continue

        valid_vectors += 1
        response_row = row_dict.copy()
        response_row.pop(vector_column, None)
        matches.append(
            {
                "similarity": similarity,
                "distance": 1 - similarity,
                "row": response_row,
            }
        )

    matches.sort(key=lambda item: item["similarity"], reverse=True)
    top_matches = matches[: request.limit]

    return SearchResponse(
        query=request.query,
        searched_rows=len(rows),
        valid_vectors=valid_vectors,
        results=[
            SearchResult(rank=index + 1, **match)
            for index, match in enumerate(top_matches)
        ],
    )


@app.post("/search", response_model=SearchResponse)
def search_embeddings(request: SearchRequest):
    return run_embedding_search(request)


@app.get("/medicine/{medicine_name}", response_model=SearchResponse)
def search_medicine(
    medicine_name: str,
    database: str = DB_NAME,
    schema_name: str = DB_SCHEMA,
    table_name: str = "documents",
    text_column: str = "content",
    vector_column: str = "embedding",
    limit: int = Query(default=5, ge=1, le=50),
    embedding_source: str = "Ollama",
    embedding_model: str = "nomic-embed-text",
    ollama_base_url: str | None = DEFAULT_OLLAMA_BASE_URL,
):
    request = SearchRequest(
        query=medicine_name,
        database=database,
        schema_name=schema_name,
        table_name=table_name,
        text_column=text_column,
        vector_column=vector_column,
        limit=limit,
        embedding_source=embedding_source,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
    )
    return run_embedding_search(request)

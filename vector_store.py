"""
vector_store.py - The vector database (ChromaDB with an HNSW index).

What lives here:
  * get_embeddings()   - turns text into vectors
  * get_vector_store() - opens (or creates) the Chroma collection
  * search()           - three search modes: similarity, threshold, mmr
  * is_duplicate()     - checks if a document is already stored
  * reset_database()   - empties the collection (used by `ingest.py --reset`)
"""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config

SEARCH_MODES = ["similarity", "threshold", "mmr"]


@lru_cache(maxsize=1)
def get_embeddings():
    """Return the embedding model chosen in config.py."""
    if config.EMBEDDING_PROVIDER == "local":
        # Free, open-source, 384-dimensional vectors.
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


def make_store(collection_name: str = config.COLLECTION_NAME, persist: bool = True) -> Chroma:
    """Create a Chroma collection. persist=False keeps it in memory (used by chunk_experiment.py)."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_DIR if persist else None,
        collection_metadata=config.HNSW_SETTINGS,   # HNSW index settings
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Open the saved Chroma collection (it is created on first use)."""
    return make_store()


def reset_database() -> None:
    """Delete every chunk and start with an empty collection."""
    get_vector_store().delete_collection()
    get_vector_store.cache_clear()


def search(query: str, k: int = config.DEFAULT_K, mode: str = "similarity",
           db: Chroma | None = None) -> list[Document]:
    """
    Search the vector DB.

    mode = "similarity" -> plain top-k nearest neighbours by cosine similarity (fast,
                           may return near-copies of the same point)
    mode = "threshold"  -> semantic search that only keeps chunks that are truly relevant;
                           returns nothing if nothing is close enough (helps "I don't know")
    mode = "mmr"        -> Maximal Marginal Relevance: relevant AND diverse chunks,
                           so you don't get 4 chunks saying the same thing
    """
    db = db or get_vector_store()

    if mode == "mmr":
        return db.max_marginal_relevance_search(query, k=k, fetch_k=k * 4)

    results = db.similarity_search_with_relevance_scores(query, k=k)
    docs = []
    for doc, score in results:
        if mode == "threshold" and score < config.RELEVANCE_THRESHOLD:
            continue
        doc.metadata["score"] = round(score, 3)
        docs.append(doc)
    return docs


def count_chunks() -> int:
    """How many chunks are stored (shown in the UI)."""
    return get_vector_store()._collection.count()


def hash_exists(file_hash: str) -> bool:
    """Is a file with exactly this fingerprint already stored?"""
    return bool(get_vector_store().get(where={"file_hash": file_hash}, limit=1)["ids"])


def is_duplicate(text: str, file_hash: str | None = None) -> tuple[bool, str]:
    """
    Check whether a document is already in the DB, in two steps:
      1. Exact check  - same file hash already stored?        (same file uploaded twice)
      2. Fuzzy check  - is the opening text nearly identical?  (same paper, different file)
    Returns (is_duplicate, reason).
    """
    db = get_vector_store()

    if file_hash and hash_exists(file_hash):
        return True, "Exact same file is already in the database."

    if count_chunks() == 0:
        return False, "Database is empty."

    # The opening of a document (title + abstract) is enough to recognise it.
    # We use one chunk's worth so it compares like-for-like with the stored first chunk.
    sample = text[:config.CHUNK_SIZE]
    results = db.similarity_search_with_relevance_scores(sample, k=1)
    if results and results[0][1] >= config.DUPLICATE_THRESHOLD:
        source = results[0][0].metadata.get("source", "unknown")
        return True, f"Very similar content already stored (from: {source})."

    return False, "No duplicate found."

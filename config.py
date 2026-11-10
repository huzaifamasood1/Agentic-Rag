"""
config.py - All settings in one place.

Change values here (or in .env) instead of hunting through the code.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")  # reads OPENAI_API_KEY (and anything else) from .env

# ---- Topic ------------------------------------------------------------------
# The knowledge base is ABOUT Agentic RAG: what RAG is, how agents plan, retrieve,
# reflect and use tools, vector databases, embeddings, prompting, evaluation.
TOPIC = "Agentic RAG (retrieval-augmented generation, LLM agents and the techniques behind them)"

# ---- Models -------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if OPENAI_API_KEY in ("", "your_key_here", "your_openai_api_key_here"):
    OPENAI_API_KEY = ""
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")        # cheap + fast, good for a demo
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")    # describes images (it can read pictures)

# "openai" -> text-embedding-3-small (1536 numbers per vector, needs the API key)
# "local"  -> all-MiniLM-L6-v2 (384 numbers per vector, free, needs sentence-transformers)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

# ---- Vector database (ChromaDB) -------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))
# One collection per embedding provider: OpenAI and MiniLM vectors have different
# lengths (1536 vs 384), so they can never share a collection.
COLLECTION_NAME = f"agentic_rag_{EMBEDDING_PROVIDER}"

# Chroma indexes vectors with HNSW (Hierarchical Navigable Small World graph).
# We chose HNSW over IVF-PQ because users can add documents at any time:
# HNSW handles growing data without retraining and keeps full-precision vectors.
# NOTE: these settings are fixed when the collection is first created.
# To change them, delete the chroma_db folder and load the data again.
HNSW_SETTINGS = {
    "hnsw:space": "cosine",        # cosine distance works best for text embeddings
    "hnsw:construction_ef": 200,   # build quality (higher = better graph, slower insert)
    "hnsw:M": 16,                  # links per node (higher = more accurate, more memory)
    "hnsw:search_ef": 100,         # search quality (higher = more accurate, slower query)
}

# ---- Chunking ---------------------------------------------------------------------
# Chosen with chunk_experiment.py (see README "Chunking experiment").
# Change them BEFORE loading data, because chunks are cut when data is added.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ---- Retrieval ------------------------------------------------------------------------
DEFAULT_K = int(os.getenv("RETRIEVER_K", "4"))   # how many chunks to fetch
RELEVANCE_THRESHOLD = 0.3      # for "threshold" search mode (0 = anything, 1 = identical)
DUPLICATE_THRESHOLD = 0.92     # above this, a new document counts as "already in the DB"

# ---- Agent ---------------------------------------------------------------------------
MAX_SEARCH_ROUNDS = 3          # first search + up to 2 "search again" rounds
MAX_CONTEXT_DOCS = 10          # cap on sources sent to the LLM

# ---- Images ----------------------------------------------------------------------------
MAX_IMAGES_PER_PDF = 4         # figures taken from each PDF and described (each costs one API call)
MIN_IMAGE_SIDE = 200           # skip icons/logos smaller than this many pixels

# ---- External APIs -------------------------------------------------------------------------
HTTP_HEADERS = {"User-Agent": "AgenticRAG-StudentProject/1.0 (university course project)"}
os.environ.setdefault("USER_AGENT", HTTP_HEADERS["User-Agent"])   # stops a LangChain web-loader warning

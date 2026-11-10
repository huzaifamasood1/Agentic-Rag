"""
tools.py - The four places the agent can search.

  notes     -> our own ChromaDB knowledge base (team's PDFs, web articles, videos, images)
  arxiv     -> research papers on RAG / LLM agents (arXiv API, free, no key)
  wikipedia -> short definitions of concepts (Wikipedia API, free, no key)
  code      -> runs a small Python calculation in a separate process

Every tool returns a list of LangChain Documents with a "source" in the metadata,
so the answer step can cite them the same way.
"""
import ast
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from functools import lru_cache

import requests
from langchain_core.documents import Document

import config
import vector_store

TIMEOUT = 15  # seconds to wait for an API


# ---------------------------------------------------------------------------
# 1. Our own notes (vector DB)
# ---------------------------------------------------------------------------
def search_notes(query: str, k: int = config.DEFAULT_K, mode: str = "similarity") -> list[Document]:
    if vector_store.count_chunks() == 0:
        return []
    docs = vector_store.search(query, k=k, mode=mode)
    for d in docs:
        d.metadata["tool"] = "notes"
    return docs


# ---------------------------------------------------------------------------
# 2. arXiv (research papers)
# ---------------------------------------------------------------------------
ARXIV_URL = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
_last_arxiv_call = [0.0]


def _arxiv_request(search_query: str, max_results: int) -> str:
    # arXiv asks for at most one request every 3 seconds, otherwise it blocks you.
    wait = 3 - (time.time() - _last_arxiv_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_arxiv_call[0] = time.time()
    response = requests.get(
        ARXIV_URL,
        params={"search_query": search_query, "max_results": max_results, "sortBy": "relevance"},
        headers=config.HTTP_HEADERS, timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.text


@lru_cache(maxsize=128)   # same query twice -> no second request
def _search_arxiv_cached(query: str, max_results: int) -> tuple:
    words = [w.strip('"?,.()') for w in query.split() if len(w.strip('"?,.()')) > 2][:5]
    if not words:
        return ()
    # First try: every word must appear. If that finds nothing, any word may appear.
    xml_text = _arxiv_request(" AND ".join(f"all:{w}" for w in words), max_results)
    entries = ET.fromstring(xml_text).findall("a:entry", ATOM)
    if not entries:
        xml_text = _arxiv_request(" OR ".join(f"ti:{w}" for w in words), max_results)
        entries = ET.fromstring(xml_text).findall("a:entry", ATOM)

    results = []
    for entry in entries:
        title = " ".join((entry.findtext("a:title", "", ATOM)).split())
        abstract = " ".join((entry.findtext("a:summary", "", ATOM)).split())
        year = (entry.findtext("a:published", "", ATOM))[:4]
        url = entry.findtext("a:id", "", ATOM)
        authors = [a.findtext("a:name", "", ATOM) for a in entry.findall("a:author", ATOM)]
        first_author = authors[0].split()[-1] + (" et al." if len(authors) > 1 else "") if authors else ""
        if abstract:
            results.append((title, abstract, year, url, first_author))
    return tuple(results)


def search_arxiv(query: str, max_results: int = 3) -> list[Document]:
    return [
        Document(
            page_content=f"{title}\n{abstract}",
            metadata={"tool": "arxiv", "source": f"arXiv: {title} ({author}, {year})", "url": url},
        )
        for title, abstract, year, url, author in _search_arxiv_cached(query, max_results)
    ]


# ---------------------------------------------------------------------------
# 3. Wikipedia (definitions of concepts)
# ---------------------------------------------------------------------------
WIKI_URL = "https://en.wikipedia.org/w/api.php"


def search_wikipedia(query: str, max_results: int = 2) -> list[Document]:
    data = requests.get(
        WIKI_URL,
        params={
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": query, "gsrlimit": max_results,
            "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exlimit": "max",
            "inprop": "url",
        },
        headers=config.HTTP_HEADERS, timeout=TIMEOUT,
    ).json()

    pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    docs = []
    for page in pages:
        text = (page.get("extract") or "").strip()
        if not text:
            continue
        docs.append(Document(
            page_content=f"{page['title']}\n{text[:3000]}",
            metadata={"tool": "wikipedia", "source": f"Wikipedia: {page['title']}",
                      "url": page.get("fullurl", "")},
        ))
    return docs


# ---------------------------------------------------------------------------
# 4. Code sandbox (calculations)
# ---------------------------------------------------------------------------
# Only these modules may be imported. Everything else (os, pathlib, socket, ...) is refused.
ALLOWED_MODULES = {"math", "statistics", "random", "itertools", "functools", "collections",
                   "fractions", "decimal", "json", "re", "datetime"}
BLOCKED_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "globals",
                 "locals", "vars", "getattr", "setattr", "delattr", "breakpoint", "exit", "quit"}


def check_code(code: str) -> str | None:
    """Return a reason if the code is not allowed, or None if it looks safe."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return f"syntax error ({error.msg})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            names = []
        for name in names:
            if name not in ALLOWED_MODULES:
                return f"import of '{name}' is not allowed"
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            return f"use of '{node.id}' is not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "access to hidden attributes (__...__) is not allowed"
    return None


def run_code(code: str) -> list[Document]:
    """
    Run LLM-written Python in a separate process with a time limit.
    The code is checked first (only maths-type imports allowed, no files, no eval).
    This is a reasonable safety layer for a class demo, not a full security sandbox.
    """
    code = code.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()

    problem = check_code(code)
    if problem:
        output = f"Code was blocked: {problem}."
    else:
        with tempfile.TemporaryDirectory() as folder:
            try:
                result = subprocess.run(
                    [sys.executable, "-I", "-c", code],
                    capture_output=True, text=True, timeout=10, cwd=folder,
                )
                output = result.stdout or result.stderr
            except subprocess.TimeoutExpired:
                output = "Code took too long and was stopped."

    return [Document(
        page_content=f"Code:\n{code}\n\nOutput:\n{output.strip()}",
        metadata={"tool": "code", "source": "Python calculation (code sandbox)"},
    )]

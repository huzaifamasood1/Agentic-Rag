"""
self_test.py - Check that the whole system works WITHOUT an OpenAI key.

Run with:   python self_test.py

A pretend AI (a "fake LLM") replaces OpenAI, and simple fake embeddings replace
the real ones, so this costs nothing and needs no key. It tests:
  1. Vector database (HNSW settings, adding chunks, 3 search modes)
  2. Ingestion (PDF pages kept, Markdown sections kept, YouTube links understood)
  3. Duplicate check
  4. Code sandbox (safe code runs, dangerous code is blocked)
  5. The full agent loop: plan -> search -> "search again" -> answer
  6. Real arXiv + Wikipedia searches (needs internet, no key)
Uses a temporary database, so your real chroma_db folder is never touched.
"""
import os
import tempfile
from pathlib import Path

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or "not-needed-for-self-test"

import config
config.CHROMA_DIR = tempfile.mkdtemp()          # throwaway database

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
import vector_store
vector_store.get_embeddings = lambda: DeterministicFakeEmbedding(size=64)

import agent
import ingest
import tools

passed, failed = 0, 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def tiny_pdf(path, pages):
    """Write a minimal PDF with one line of text per page (no extra library needed)."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objects)} 0 R >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = "%PDF-1.4\n", []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out.encode()))
        out += f"{number} 0 obj\n{body}\nendobj\n"
    xref = len(out.encode())
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    Path(path).write_bytes(out.encode())


# ---- A pretend AI that gives scripted decisions ------------------------------
class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    decisions = []   # the Plan / Reflection objects it will return, in order

    def with_structured_output(self, schema):
        class Structured:
            def invoke(self, prompt):
                return FakeLLM.decisions.pop(0)
        return Structured()

    def invoke(self, prompt):
        if "Python script" in prompt:
            return FakeMessage("import math\nprint('cosine:', round(25 / math.sqrt(14 * 45), 3))")
        return FakeMessage("Agentic RAG adds planning and reflection to RAG [1]. (pretend answer)")


agent.get_llm = lambda temperature=0: FakeLLM()


# ---- 1. Vector database ---------------------------------------------------------
print("\n1. Vector database")
db = vector_store.get_vector_store()
check("HNSW uses cosine distance", db._collection.metadata.get("hnsw:space") == "cosine")

paper = ("# Self-RAG\n\n## Reflection tokens\n\n"
         + "Self-RAG generates reflection tokens to decide when to retrieve. " * 30)
added = ingest.add_text(paper, {"source": "self_rag.pdf", "file_hash": "hash123"}, markdown=True)
ingest.add_text("HNSW is a graph index for approximate nearest neighbour search. " * 40,
                {"source": "hnsw_article"})
check(f"Paper split into {added} chunks and stored", vector_store.count_chunks() > added)

for mode in vector_store.SEARCH_MODES:
    try:
        results = vector_store.search("reflection tokens", k=3, mode=mode)
        check(f"Search mode '{mode}' runs ({len(results)} results)", True)
    except Exception as error:
        check(f"Search mode '{mode}' runs", False, error)

# ---- 2. Ingestion ------------------------------------------------------------------
print("\n2. Ingestion")
chunks = ingest.make_chunks([{"text": paper, "metadata": {"source": "x"}, "markdown": True}])
check("Markdown chunks remember their section title",
      chunks and chunks[0].metadata.get("section") == "Reflection tokens", chunks[0].metadata if chunks else "")

pdf_path = Path(tempfile.mkdtemp()) / "test.pdf"
tiny_pdf(pdf_path, ["Page one is about naive RAG.", "Page two is about agentic RAG."])
pages, _ = ingest.load_pdf(str(pdf_path), use_marker=False)
check("PDF pages keep their page number",
      [p["metadata"]["page"] for p in pages] == [1, 2] and "agentic" in pages[1]["text"],
      [p["metadata"] for p in pages])

check("YouTube links are understood",
      ingest.youtube_id("https://www.youtube.com/watch?v=0z9_MhcYvcY") == "0z9_MhcYvcY"
      and ingest.youtube_id("https://youtu.be/0z9_MhcYvcY") == "0z9_MhcYvcY")

# ---- 3. Duplicate check -----------------------------------------------------------
print("\n3. Duplicate check")
is_dup, reason = vector_store.is_duplicate("any text", "hash123")
check("Same file is rejected", is_dup, reason)
is_dup, reason = vector_store.is_duplicate("A brand new article about query rewriting. " * 20, "new_hash")
check("New file is accepted", not is_dup, reason)

# ---- 4. Code sandbox -------------------------------------------------------------
print("\n4. Code sandbox")
output = tools.run_code("print(2 + 2)")[0].page_content
check("Runs safe code (2 + 2 = 4)", "4" in output, output)
for name, code in [("import os", "import os\nos.remove('x')"),
                   ("from os import ...", "from os import getcwd\nprint(getcwd())"),
                   ("pathlib", "import pathlib\nprint(list(pathlib.Path('/').iterdir()))"),
                   ("open()", "print(open('/etc/passwd').read())")]:
    output = tools.run_code(code)[0].page_content
    check(f"Blocks dangerous code ({name})", "blocked" in output, output)

# ---- 5. Agent loop -----------------------------------------------------------------
print("\n5. Agent decisions")
FakeLLM.decisions = [agent.Plan(needs_retrieval=False, queries=[], sources=[], reasoning="Just a greeting")]
result = agent.ask("hi")
check("Skips searching for 'hi'", "Decided no search is needed" in result["trace"])

FakeLLM.decisions = [
    agent.Plan(needs_retrieval=True, queries=["agentic rag"], sources=["notes"],
               reasoning="Needs facts about agentic RAG"),
    agent.Reflection(sufficient=False, reason="These results are about chemical agents",
                     new_queries=["LLM agent retrieval"], new_sources=["notes"]),
    agent.Reflection(sufficient=True, reason="Now covered", new_queries=[], new_sources=[]),
]
result = agent.ask("what is agentic rag?", style="Chain of Thought")
check("Rewrites the question into queries", any("Rewrote" in t for t in result["trace"]))
check("Searches again after bad results", any("Searching again" in t for t in result["trace"]))
check("Answer has a citation", "[1]" in result["answer"])

# The "search again" results must survive the MAX_CONTEXT_DOCS cut, even if round 1 found a lot
original_search = tools.search_notes
tools.search_notes = lambda query, k=4, mode="similarity": (
    [Document(page_content="GOOD: LLM agents plan and retrieve", metadata={"source": "good"})]
    if query == "better query" else
    [Document(page_content=f"weak result {query} {i}", metadata={"source": "weak"}) for i in range(12)])
FakeLLM.decisions = [
    agent.Plan(needs_retrieval=True, queries=["q1"], sources=["notes"], reasoning="search"),
    agent.Reflection(sufficient=False, reason="weak", new_queries=["better query"], new_sources=["notes"]),
    agent.Reflection(sufficient=True, reason="ok", new_queries=[], new_sources=[]),
]
result = agent.ask("test")
tools.search_notes = original_search
check("Improved results from 'search again' reach the answer step",
      any("GOOD" in d.page_content for d in result["sources"]))

FakeLLM.decisions = [agent.Plan(needs_retrieval=True, queries=["cosine"], sources=["code"],
                                reasoning="This is a calculation"),
                     agent.Reflection(sufficient=True, reason="Done", new_queries=[], new_sources=[])]
result = agent.ask("cosine similarity of [1,2,3] and [2,4,5]?")
check("Uses the code sandbox for maths", "0.996" in result["sources"][0].page_content,
      result["sources"][0].page_content)

check("Final answer is extracted from Chain of Thought output",
      agent.final_answer("Step 1 ... Final answer: RAG retrieves [1].") == "RAG retrieves [1].")

print("\n   What the agent did (example):")
for step in result["trace"]:
    print(f"     - {step}")

# ---- 6. Real external sources (internet, no key) -----------------------------------
print("\n6. Real arXiv + Wikipedia (needs internet)")
try:
    papers = tools.search_arxiv("agentic retrieval augmented generation", max_results=2)
    check(f"arXiv returned {len(papers)} papers", len(papers) > 0)
    if papers:
        print(f"     e.g. {papers[0].metadata['source']}")
except Exception as error:
    check("arXiv reachable", False, f"({type(error).__name__} - check your internet)")

try:
    pages = tools.search_wikipedia("retrieval-augmented generation")
    check(f"Wikipedia returned {len(pages)} pages", len(pages) > 0)
    if pages:
        print(f"     e.g. {pages[0].metadata['source']}")
except Exception as error:
    check("Wikipedia reachable", False, f"({type(error).__name__} - check your internet)")

# ---- Summary -------------------------------------------------------------------------
print(f"\n===== {passed} passed, {failed} failed =====")
if failed == 0:
    print("Everything works. Add the real OpenAI key to .env and run:  streamlit run app.py")

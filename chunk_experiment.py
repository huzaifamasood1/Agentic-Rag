"""
chunk_experiment.py - Find the best chunk size and overlap for OUR documents.

The guidelines ask us to "experiment with different chunk sizes and overlap, and find the
optimum value for your document types". This script does exactly that:

  1. Loads the team's documents from data/ (PDFs, web articles, videos, saved texts).
  2. For every (chunk size, overlap) pair, builds a temporary in-memory database.
  3. Asks every in_scope test question from tests/evaluation_cases.json and checks whether
     the retrieved chunks contain the expected keywords.
  4. Prints a table and saves evaluation_results/chunk_experiment.csv.

Run with:   python chunk_experiment.py                 (pypdf for PDFs: fast)
            python chunk_experiment.py --marker        (Marker for PDFs: slow, keeps headings)
            python chunk_experiment.py --by-type       (also show results per document type)
Cost: only embeddings (a few cents with text-embedding-3-small). No chat model is used.
Your real chroma_db is NOT touched. Put the winner into CHUNK_SIZE / CHUNK_OVERLAP in config.py.

Metrics:
  keyword recall  - share of expected keywords found in the top-k chunks (higher = better)
  hit rate        - share of questions where at least one keyword was found
  MRR             - 1 / rank of the first chunk containing a keyword (higher = found earlier)
  context chars   - how much text is sent to the LLM (lower = cheaper, less noise)
"""
import argparse
import csv
import json
import statistics
import time
import uuid

import config
import ingest
import vector_store
from load_team_data import DATA, read_list

SETTINGS = [(300, 50), (500, 75), (800, 120), (1000, 150), (1500, 200), (2000, 300)]


def load_raw_documents(use_marker: bool) -> list[dict]:
    raw = []
    for pdf in sorted((DATA / "pdfs").glob("*.pdf")):
        pages, _ = ingest.load_pdf(str(pdf), use_marker=use_marker)
        for page in pages:
            page["metadata"]["type"] = "pdf"
        raw += pages
        print(f"  loaded {pdf.name} ({len(pages)} pages)")
    for url in read_list(DATA / "urls.txt"):
        try:
            docs = ingest.load_web(url)
            docs[0]["metadata"]["type"] = "web"
            raw += docs
            print(f"  loaded {url}")
        except Exception as error:
            print(f"  skipped {url}: {error}")
    for url in read_list(DATA / "youtube.txt"):
        try:
            docs = ingest.load_youtube(url)
            docs[0]["metadata"]["type"] = "youtube"
            raw += docs
            print(f"  loaded {url}")
        except Exception as error:
            print(f"  skipped {url}: {error}")
    for txt in sorted((DATA / "texts").glob("*.txt")):
        raw.append({"text": txt.read_text(encoding="utf-8"), "markdown": False,
                    "metadata": {"source": txt.name, "type": "text"}})
    return [r for r in raw if r["text"].strip()]


def score(db, cases, k):
    recalls, hits, ranks, chars, seconds = [], [], [], [], []
    for case in cases:
        keywords = [w.lower() for w in case["expected_keywords"]]
        start = time.time()
        docs = vector_store.search(case["question"], k=k, mode="similarity", db=db)
        seconds.append(time.time() - start)
        texts = [d.page_content.lower() for d in docs]
        joined = " ".join(texts)
        found = [w for w in keywords if w in joined]
        recalls.append(len(found) / len(keywords))
        hits.append(bool(found))
        first = next((i for i, t in enumerate(texts, start=1) if any(w in t for w in keywords)), None)
        ranks.append(1 / first if first else 0)
        chars.append(sum(len(t) for t in texts))
    return {
        "keyword_recall": round(statistics.mean(recalls), 3),
        "hit_rate": round(statistics.mean(hits), 3),
        "mrr": round(statistics.mean(ranks), 3),
        "avg_context_chars": int(statistics.mean(chars)),
        "avg_query_seconds": round(statistics.mean(seconds), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare chunk sizes on the team's data")
    parser.add_argument("--k", type=int, default=config.DEFAULT_K)
    parser.add_argument("--marker", action="store_true", help="Use Marker for PDFs (slow)")
    parser.add_argument("--by-type", action="store_true", help="Also test each document type alone")
    args = parser.parse_args()

    cases = [c for c in json.loads((config.PROJECT_ROOT / "tests" / "evaluation_cases.json").read_text("utf-8"))
             if c["category"] == "in_scope" and c.get("expected_keywords")]
    print("Loading documents from data/ ...")
    raw = load_raw_documents(args.marker)
    if not raw or not cases:
        raise SystemExit("Need documents in data/ and in_scope questions with expected_keywords.")

    groups = {"all": raw}
    if args.by_type:
        for doc_type in sorted({r["metadata"]["type"] for r in raw}):
            groups[doc_type] = [r for r in raw if r["metadata"]["type"] == doc_type]

    rows = []
    for group, docs in groups.items():
        for size, overlap in SETTINGS:
            chunks = ingest.make_chunks(docs, chunk_size=size, chunk_overlap=overlap)
            db = vector_store.make_store(collection_name=f"experiment_{uuid.uuid4().hex[:8]}", persist=False)
            start = time.time()
            for i in range(0, len(chunks), 500):          # add in batches
                db.add_documents(chunks[i:i + 500])
            build = time.time() - start
            row = {"documents": group, "chunk_size": size, "chunk_overlap": overlap,
                   "chunks": len(chunks), "index_seconds": round(build, 1), **score(db, cases, args.k)}
            db.delete_collection()
            rows.append(row)
            print(f"  [{group}] size {size:>4} / overlap {overlap:>3}: {len(chunks):>5} chunks | "
                  f"recall {row['keyword_recall']:.2f} | hit {row['hit_rate']:.2f} | MRR {row['mrr']:.2f} | "
                  f"{row['avg_context_chars']} chars sent")

    config_path = config.PROJECT_ROOT / "evaluation_results"
    config_path.mkdir(exist_ok=True)
    with open(config_path / "chunk_experiment.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== BEST SETTING PER DOCUMENT GROUP =====")
    for group in groups:
        mine = [r for r in rows if r["documents"] == group]
        # Best recall; ties broken by MRR, then by sending less text to the LLM
        best = max(mine, key=lambda r: (r["keyword_recall"], r["mrr"], -r["avg_context_chars"]))
        print(f"  {group:<8}: CHUNK_SIZE={best['chunk_size']}, CHUNK_OVERLAP={best['chunk_overlap']} "
              f"(recall {best['keyword_recall']}, MRR {best['mrr']})")
    print(f"\nSaved {config_path / 'chunk_experiment.csv'}. Put the 'all' winner into config.py, "
          f"delete chroma_db/ and run load_team_data.py again.")


if __name__ == "__main__":
    main()

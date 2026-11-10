"""
download_papers.py - Download the core Agentic RAG research papers from arXiv into data/pdfs/.

Run with:   python download_papers.py
Already-downloaded papers are skipped. Edit PAPERS to add or remove papers.
All papers are freely available on arXiv.
"""
import time

import requests

import config

PAPERS = {
    # --- RAG foundations ---
    "2005.11401": "rag_lewis_2020",                 # Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
    "2312.10997": "rag_survey_gao_2023",            # Retrieval-Augmented Generation for Large Language Models: A Survey
    "2004.04906": "dense_passage_retrieval_2020",   # Dense Passage Retrieval (DPR)
    # --- Agentic RAG ---
    "2501.09136": "agentic_rag_survey_2025",        # Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG
    "2310.11511": "self_rag_2023",                  # Self-RAG: retrieve, generate and critique through self-reflection
    "2401.15884": "corrective_rag_2024",            # Corrective RAG (CRAG)
    "2403.14403": "adaptive_rag_2024",              # Adaptive-RAG: routing by question complexity
    "2305.06983": "flare_active_rag_2023",          # Active Retrieval Augmented Generation (FLARE)
    # --- Agents, tools and prompting ---
    "2210.03629": "react_2022",                     # ReAct: reasoning + acting
    "2302.04761": "toolformer_2023",                # Toolformer: LLMs teach themselves to use tools
    "2201.11903": "chain_of_thought_2022",          # Chain-of-Thought prompting
    "2305.10601": "tree_of_thoughts_2023",          # Tree of Thoughts
    "2212.10496": "hyde_2022",                      # HyDE: hypothetical document embeddings (query rewriting)
    # --- Vector search and evaluation ---
    "1603.09320": "hnsw_2016",                      # HNSW graph index (what ChromaDB uses)
    "2309.15217": "ragas_2023",                     # RAGAS: automated evaluation of RAG
}


def main():
    folder = config.PROJECT_ROOT / "data" / "pdfs"
    folder.mkdir(parents=True, exist_ok=True)
    for arxiv_id, name in PAPERS.items():
        path = folder / f"{name}.pdf"
        if path.exists():
            print(f"  {path.name}: already downloaded")
            continue
        try:
            response = requests.get(f"https://arxiv.org/pdf/{arxiv_id}",
                                    headers=config.HTTP_HEADERS, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
            print(f"  {path.name}: downloaded ({len(response.content) // 1024} KB)")
        except Exception as error:
            print(f"  {path.name}: ERROR {error}")
        time.sleep(3)   # arXiv asks for a pause between downloads
    print(f"\nDone. Now run:  python load_team_data.py")


if __name__ == "__main__":
    main()

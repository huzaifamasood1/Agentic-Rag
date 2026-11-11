"""
evaluate.py - Compare the plain LLM with our agentic RAG on a fixed set of test questions.

Run with:
    python evaluate.py                               # Zero-shot + similarity search
    python evaluate.py --styles all                  # compare the 4 prompt styles
    python evaluate.py --modes all                   # compare the 3 search modes
    python evaluate.py --styles all --modes all      # everything (12 RAG runs per question)

Questions come from tests/evaluation_cases.json. Each has a category:
    in_scope      - answerable from our sources; has a reference answer
    unanswerable  - on-topic but invented (fake paper, false premise) -> a hallucination trap
    out_of_scope  - not about Agentic RAG -> our assistant should decline

What we measure (an LLM judge compares each answer with the reference answer):
    * accuracy        - in_scope answers that are correct
    * hallucination   - answers with invented or false facts; for unanswerable questions,
                        ANY confident answer counts as a hallucination
    * refusal         - "I don't know" / declining, on unanswerable + out_of_scope questions
    * relevance       - does the answer address the question (1-5)
    * faithfulness    - RAG only: is every claim supported by the retrieved sources?
    * response time   - seconds per answer
Results go to evaluation_results/ (open the CSVs in Excel for the slides).

Note: the judge is an LLM (gpt-4o-mini via the OpenAI API by default; set JUDGE_MODEL in .env to change), so scores are an automatic estimate.
Check a sample by hand for the presentation.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import agent
import config
import prompts
import vector_store

CASES_FILE = config.PROJECT_ROOT / "tests" / "evaluation_cases.json"
OUTPUT = config.PROJECT_ROOT / "evaluation_results"


class Judgement(BaseModel):
    correct: bool = Field(description="Does the answer agree with the reference answer?")
    hallucinated: bool = Field(description="Does the answer state invented or false facts?")
    declined: bool = Field(description="Does the answer decline, or say it does not know?")
    relevance: int = Field(description="1-5: how well the answer addresses the question")
    reason: str = Field(description="One sentence")


class Faithfulness(BaseModel):
    faithful: bool
    reason: str


JUDGE_PROMPT = """You are grading an answer from an AI assistant about Agentic RAG.

QUESTION: {question}
CATEGORY: {category}
REFERENCE ANSWER: {reference}

ANSWER TO GRADE:
{answer}

Rules:
- correct: true if the answer gives the same key facts as the reference (wording may differ).
  For category "unanswerable" or "out_of_scope", correct is true only if the answer declines,
  says it does not know, or points out the false premise.
- hallucinated: true if the answer states facts that are invented or contradict the reference.
  For category "unanswerable", any confident answer about the invented thing is a hallucination.
  For "out_of_scope", answering the off-topic question correctly is NOT a hallucination.
- declined: true if the answer refuses, says "I don't know", or says it focuses on another topic.
- relevance: 1 = ignores the question, 5 = fully addresses it (a correct refusal counts as 5)."""

FAITHFUL_PROMPT = """Check if the ANSWER only uses facts found in the SOURCES.
Set faithful to false if the answer adds facts that are not in the sources.
An answer that says "I don't know" is faithful.

SOURCES:
{sources}

ANSWER:
{answer}"""


def get_judge_llm():
    if not config.OPENAI_API_KEY:
        raise SystemExit("The judge uses the OpenAI API: set OPENAI_API_KEY in .env")
    return ChatOpenAI(model=config.JUDGE_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)


def judge(question, category, reference, answer) -> Judgement:
    return get_judge_llm().with_structured_output(Judgement).invoke(JUDGE_PROMPT.format(
        question=question, category=category, reference=reference, answer=answer))


def is_faithful(answer, sources):
    if not sources:
        return None  # nothing to compare against
    text = "\n\n".join(d.page_content for d in sources)
    return get_judge_llm().with_structured_output(Faithfulness).invoke(
        FAITHFUL_PROMPT.format(sources=text[:30000], answer=answer)).faithful


def percent(rows, key, category=None):
    chosen = [r for r in rows if (category is None or r["category"] in category) and r[key] is not None]
    return round(100 * sum(bool(r[key]) for r in chosen) / len(chosen), 1) if chosen else None


def summarise(name, rows):
    times = [r["seconds"] for r in rows]
    return {
        "system": name,
        "questions": len(rows),
        "accuracy_in_scope_%": percent(rows, "correct", {"in_scope"}),
        "hallucination_all_%": percent(rows, "hallucinated"),
        "hallucination_unanswerable_%": percent(rows, "hallucinated", {"unanswerable"}),
        "refusal_unanswerable+out_of_scope_%": percent(rows, "declined", {"unanswerable", "out_of_scope"}),
        "false_refusal_in_scope_%": percent(rows, "declined", {"in_scope"}),
        "avg_relevance_1to5": round(statistics.mean(r["relevance"] for r in rows), 2),
        "faithfulness_%": percent(rows, "faithful"),
        "avg_seconds": round(statistics.mean(times), 2),
        "median_seconds": round(statistics.median(times), 2),
        "avg_sources": round(statistics.mean(r["sources"] for r in rows), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Plain LLM vs agentic RAG evaluation")
    parser.add_argument("--cases", default=str(CASES_FILE), help="JSON file with test questions")
    parser.add_argument("--styles", default="Tree of Thoughts",
                        help=f"Comma-separated prompt styles or 'all' ({', '.join(prompts.ANSWER_PROMPTS)})")
    parser.add_argument("--modes", default="similarity",
                        help=f"Comma-separated search modes or 'all' ({', '.join(vector_store.SEARCH_MODES)})")
    parser.add_argument("--k", type=int, default=config.DEFAULT_K)
    parser.add_argument("--limit", type=int, help="Only use the first N questions (quick test)")
    args = parser.parse_args()

    styles = list(prompts.ANSWER_PROMPTS) if args.styles == "all" else [s.strip() for s in args.styles.split(",")]
    modes = vector_store.SEARCH_MODES if args.modes == "all" else [m.strip() for m in args.modes.split(",")]
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))[:args.limit]
    print(f"{len(cases)} questions | styles: {styles} | modes: {modes} | k={args.k} | "
          f"chunks in database: {vector_store.count_chunks()}")

    rows = []
    for number, case in enumerate(cases, start=1):
        question, category = case["question"], case["category"]
        reference = case.get("reference_answer", "")
        print(f"\n[{number}/{len(cases)}] {question}")

        # BEFORE: plain LLM (asked once per question)
        base = agent.ask_base_llm(question)
        verdict = judge(question, category, reference, base["answer"])
        rows.append({"system": "plain LLM", "style": "-", "mode": "-", "question": question,
                     "category": category, "answer": base["answer"], "seconds": base["seconds"],
                     "sources": 0, "searched": False, "faithful": None, **verdict.model_dump()})
        print(f"   plain LLM        correct={verdict.correct} hallucinated={verdict.hallucinated}")

        # AFTER: our agentic RAG, for every style x mode
        for style in styles:
            for mode in modes:
                try:
                    rag = agent.ask(question, style=style, k=args.k, search_mode=mode)
                except Exception as error:
                    print(f"   RAG {style}/{mode} ERROR: {error}")
                    continue
                answer = agent.final_answer(rag["answer"])
                verdict = judge(question, category, reference, answer)
                rows.append({
                    "system": f"RAG {style} / {mode}", "style": style, "mode": mode,
                    "question": question, "category": category, "answer": answer,
                    "seconds": rag["seconds"], "sources": len(rag["sources"]),
                    "searched": "Decided no search is needed" not in rag["trace"],
                    "faithful": is_faithful(answer, rag["sources"]), **verdict.model_dump(),
                })
                print(f"   RAG {style}/{mode}: correct={verdict.correct} "
                      f"hallucinated={verdict.hallucinated} ({rag['seconds']}s)")

    OUTPUT.mkdir(exist_ok=True)
    with open(OUTPUT / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    systems = list(dict.fromkeys(r["system"] for r in rows))
    summary = [summarise(name, [r for r in rows if r["system"] == name]) for name in systems]
    with open(OUTPUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print("\n===== SUMMARY (for the results slide) =====")
    for s in summary:
        print(f"{s['system']:<34} accuracy {s['accuracy_in_scope_%']}% | hallucination "
              f"{s['hallucination_all_%']}% | refusal when it should {s['refusal_unanswerable+out_of_scope_%']}% | "
              f"faithful {s['faithfulness_%']}% | {s['avg_seconds']}s avg")
    print(f"\nSaved {OUTPUT / 'results.csv'} (every answer) and {OUTPUT / 'summary.csv'} (one row per system)")


if __name__ == "__main__":
    main()

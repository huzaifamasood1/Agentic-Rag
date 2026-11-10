"""
agent.py - The agentic RAG loop. This is the "brain" of the system.

Unlike plain RAG (always search -> answer), our agent makes four decisions:

    question
       |
    1. PLAN    -> Do we need to search at all?  (no -> answer directly)
               -> Rewrite the question into better search queries
               -> Choose where to search: notes / arxiv / wikipedia / code
       |
    2. SEARCH  -> run every query on every chosen source
       |
    3. REFLECT -> Are the results good enough? ("these are about chemical agents...")
               -> if not, make new queries and go back to step 2 (max 3 rounds)
       |
    4. ANSWER  -> write the answer with [1][2] citations, or "I don't know"

Every decision is saved in `trace` so the UI can show what the agent did.
"""
import time
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
import prompts
import tools

Source = Literal["notes", "arxiv", "wikipedia", "code"]
DONT_KNOW = "I don't know based on the available sources."


# ---- What the LLM must return at each decision (structured output) ----------
class Plan(BaseModel):
    needs_retrieval: bool = Field(description="Does this question need sources?")
    queries: list[str] = Field(description="1-3 improved search queries")
    sources: list[Source] = Field(description="Where to search")
    reasoning: str = Field(description="One sentence explaining the plan")


class Reflection(BaseModel):
    sufficient: bool = Field(description="Are the results good enough to answer?")
    reason: str = Field(description="One sentence explaining why")
    new_queries: list[str] = Field(description="New queries if not sufficient, else empty")
    new_sources: list[Source] = Field(description="Sources for new queries, else empty")


def get_llm(temperature: float = 0):
    return ChatOpenAI(model=config.CHAT_MODEL, temperature=temperature)


# ---- Helpers ------------------------------------------------------------------
def run_search(queries, sources, question, k, search_mode, trace):
    """Run every query on every source. Errors are logged, not crashed on."""
    found = []
    for source in sources:
        if source == "code":
            # Code is written once for the whole question, not per query
            code = get_llm().invoke(prompts.CODE_PROMPT.format(question=question)).content
            found += tools.run_code(code)
            trace.append("Ran a Python calculation")
            continue

        for query in queries:
            try:
                if source == "notes":
                    docs = tools.search_notes(query, k=k, mode=search_mode)
                elif source == "arxiv":
                    docs = tools.search_arxiv(query)
                else:
                    docs = tools.search_wikipedia(query)
                trace.append(f"Searched {source} for '{query}' -> {len(docs)} results")
                found += docs
            except Exception as error:
                trace.append(f"Searching {source} failed ({type(error).__name__}), skipped")
    return found


def remove_duplicates(docs):
    seen, unique = set(), []
    for doc in docs:
        key = doc.page_content[:200]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def format_context(docs):
    """Number the sources so the LLM can cite them as [1], [2], ..."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        label = meta.get("source", "unknown")
        if meta.get("page") is not None:
            label += f", page {meta['page']}"
        if meta.get("section"):
            label += f", section '{meta['section']}'"
        blocks.append(f"[{i}] ({label})\n{doc.page_content}")
    return "\n\n".join(blocks)


def _result(answer, docs, trace, start):
    return {"answer": answer, "sources": docs, "trace": trace,
            "seconds": round(time.time() - start, 2)}


# ---- The main function ------------------------------------------------------------
def ask(question: str, style: str = "Zero-shot", k: int = config.DEFAULT_K,
        search_mode: str = "similarity") -> dict:
    """
    Answer a question with the agentic RAG loop.
    Returns: {"answer", "sources", "trace", "seconds"}
    """
    start = time.time()
    trace = []

    # 1. PLAN ---------------------------------------------------------------
    plan = get_llm().with_structured_output(Plan).invoke(
        prompts.ROUTER_PROMPT.format(question=question))
    trace.append(f"Plan: {plan.reasoning}")

    if not plan.needs_retrieval:
        trace.append("Decided no search is needed")
        answer = get_llm().invoke(prompts.NO_RETRIEVAL_PROMPT.format(question=question)).content
        return _result(answer, [], trace, start)

    trace.append(f"Rewrote question into: {plan.queries}")
    trace.append(f"Chose sources: {plan.sources}")

    # 2 + 3. SEARCH, then REFLECT, repeat if needed ----------------------------
    docs = []
    queries, sources = plan.queries or [question], plan.sources or ["notes"]
    for round_number in range(1, config.MAX_SEARCH_ROUNDS + 1):
        # Newest results go FIRST: a "search again" round was asked for because the
        # earlier results were weak, so its results must not be cut off by the
        # MAX_CONTEXT_DOCS limit below.
        new_docs = run_search(queries, sources, question, k, search_mode, trace)
        docs = remove_duplicates(new_docs + docs)

        if round_number == config.MAX_SEARCH_ROUNDS:
            break  # no more rounds left, answer with what we have

        reflection = get_llm().with_structured_output(Reflection).invoke(
            prompts.REFLECT_PROMPT.format(question=question,
                                          context=format_context(docs) or "(nothing found)"))
        if reflection.sufficient or not reflection.new_queries:
            trace.append(f"Results are good enough: {reflection.reason}")
            break

        trace.append(f"Searching again: {reflection.reason}")
        queries = reflection.new_queries
        sources = reflection.new_sources or sources

    # 4. ANSWER ---------------------------------------------------------------
    docs = docs[:config.MAX_CONTEXT_DOCS]
    if not docs:
        trace.append("No sources found -> 'I don't know' fallback")
        return _result(DONT_KNOW, [], trace, start)

    prompt = prompts.ANSWER_PROMPTS[style].format(context=format_context(docs), question=question)
    answer = get_llm().invoke(prompt).content
    trace.append(f"Wrote answer using {len(docs)} sources ({style} prompt)")
    return _result(answer, docs, trace, start)


def ask_base_llm(question: str) -> dict:
    """The 'before' side of the demo: the LLM alone, no retrieval, no citations."""
    start = time.time()
    answer = get_llm().invoke(prompts.BASE_LLM_PROMPT.format(question=question)).content
    return {"answer": answer, "seconds": round(time.time() - start, 2)}


def final_answer(answer: str) -> str:
    """Chain of Thought / Tree of Thoughts answers show their reasoning first.
    This returns only the part after 'Final answer:' (used by the evaluation)."""
    marker = "final answer:"
    position = answer.lower().rfind(marker)
    return answer[position + len(marker):].strip() if position >= 0 else answer

"""
prompts.py - Every prompt the system uses, kept in one file so they are easy to show
and compare in the presentation.

Prompt engineering techniques used:
  * Zero-shot          - instructions only
  * Few-shot           - instructions + worked examples
  * Chain of Thought   - think step by step before answering
  * Tree of Thoughts   - consider several explanations, then pick the best
Every answer prompt also has citation rules and an "I don't know" fallback.

Context engineering (what the LLM gets to see) happens in agent.py:
numbered sources, at most config.MAX_CONTEXT_DOCS of them, newest search round first.
"""

# --------------------------------------------------------------------------
# 1. ROUTER: decides IF to retrieve, WHAT to search and WHERE
# --------------------------------------------------------------------------
ROUTER_PROMPT = """You are the planning step of a research assistant about Agentic RAG
(retrieval-augmented generation, LLM agents, vector databases, embeddings, prompting and
evaluation of LLM systems).

Look at the user's question and decide:
1. needs_retrieval: Does it need sources?
   - false for greetings, chit-chat, or questions outside the topic
   - true for anything specific, technical, recent, or where accuracy matters
2. queries: Rewrite the question into 1-3 short, precise search queries.
   Use proper technical terms. If the question is vague, make it specific.
   Example: "how does the AI check its own answers" -> "self-reflection retrieval augmented generation",
            "Self-RAG reflection tokens"
3. sources: Pick where to search (one or more):
   - "notes"     : our own knowledge base (team's papers, articles, videos, diagrams). Usually the best first choice.
   - "arxiv"     : research papers (named methods like Self-RAG, CRAG, ReAct, recent findings, benchmarks)
   - "wikipedia" : a general definition of a concept (e.g. "what is a vector database")
   - "code"      : calculations (e.g. cosine similarity of two vectors, token or cost estimates, number of chunks)

Focus on Agentic RAG and the AI/NLP techniques around it. If the question is clearly
outside this topic (sport, cooking, celebrities ...), set needs_retrieval to false.

Question: {question}"""


# --------------------------------------------------------------------------
# 2. REFLECTION: after seeing results, should we search again?
# --------------------------------------------------------------------------
REFLECT_PROMPT = """You are checking search results for a question about Agentic RAG and LLM systems.

Question: {question}

Results found so far:
{context}

Are these results good enough to answer the question well?
Watch for common problems:
- Results use the word "agent" in a different sense (travel agents, chemical agents,
  reinforcement-learning robots) instead of LLM agents
- Results are about plain RAG when the question is about agentic RAG (or the other way round)
- Results are off-topic or too general
- An important part of the question is not covered

If the results are good enough, set sufficient to true.
If not, set sufficient to false, explain why in one sentence, and give 1-2 NEW queries
(and sources) that would fix the problem. Example: "these results are about chemical agents" ->
query "LLM agent retrieval augmented generation" on arxiv."""


# --------------------------------------------------------------------------
# 3. ANSWER prompts (one per prompting style)
# --------------------------------------------------------------------------
BASE_RULES = """You are a helpful assistant that explains Agentic RAG and related AI techniques.

RULES:
- Answer ONLY using the numbered sources below.
- Cite every fact with its source number in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply exactly:
  "I don't know based on the available sources."
  and do not make anything up.
- If sources disagree, say so and cite both.
- Keep the answer clear enough for a university student."""

ZERO_SHOT = BASE_RULES + """

Sources:
{context}

Question: {question}

Answer:"""

FEW_SHOT = BASE_RULES + """

Here are examples of good answers:

Example 1
Sources:
[1] Retrieval-augmented generation (RAG) combines a retriever that finds relevant passages with a generator that writes the answer from them.
Question: What are the two parts of a RAG system?
Answer: A RAG system has a retriever, which finds relevant passages, and a generator, which writes the answer using those passages [1].

Example 2
Sources:
[1] ChromaDB stores embeddings together with metadata such as the source file.
Question: Which company created the transformer architecture?
Answer: I don't know based on the available sources.

Now your turn.

Sources:
{context}

Question: {question}

Answer:"""

CHAIN_OF_THOUGHT = BASE_RULES + """

Think step by step. Use EXACTLY this format, one line per step, nothing before or after:
Step 1 - Relevant sources: <which source numbers matter and why>
Step 2 - Key facts: <the key facts from those sources, with citations>
Step 3 - Connection: <how the facts fit together to answer the question>
Final answer: <the answer, with citations>

Sources:
{context}

Question: {question}"""

TREE_OF_THOUGHTS = BASE_RULES + """

Use a Tree of Thoughts approach. Use EXACTLY this format, one line each, nothing before or after:
Branch A: <first possible explanation, with citations>
Support A: <strong | weak | none> - <one short reason>
Branch B: <second possible explanation, with citations>
Support B: <strong | weak | none> - <one short reason>
Branch C: <third possible explanation, with citations (optional, skip if there are only two)>
Support C: <strong | weak | none> - <one short reason>
Best: <A | B | C> - <why this branch is best supported>
Final answer: <the answer based on the best branch, with citations>

Sources:
{context}

Question: {question}"""

ANSWER_PROMPTS = {
    "Zero-shot": ZERO_SHOT,
    "Few-shot": FEW_SHOT,
    "Chain of Thought": CHAIN_OF_THOUGHT,
    "Tree of Thoughts": TREE_OF_THOUGHTS,
}

# Used when the router says "no retrieval needed"
NO_RETRIEVAL_PROMPT = """You are a friendly assistant that explains Agentic RAG and LLM systems.
Answer briefly. If the question is not about AI, RAG or LLM agents, politely say that you
focus on Agentic RAG and related AI topics.

Question: {question}"""

# Used for the "before" side of the before/after comparison (no RAG at all)
BASE_LLM_PROMPT = """Answer this question.

Question: {question}"""


# --------------------------------------------------------------------------
# 4. CODE TOOL: write Python for calculations
# --------------------------------------------------------------------------
CODE_PROMPT = """Write a short Python script that calculates the answer to this question.
Use only the standard library (math, statistics). Print the result with a clear label.
Return ONLY the code, no explanation, no markdown fences.

Question: {question}"""


# --------------------------------------------------------------------------
# 5. INGESTION CHECK: is this document worth adding?
# --------------------------------------------------------------------------
WORTH_ADDING_PROMPT = """You decide whether a document should be added to a knowledge base about
AGENTIC RAG.

Here is the start of the document:
---
{text}
---

Set worth_adding to true if it is about retrieval-augmented generation, LLM agents, tool use,
planning or reflection in LLMs, vector databases, embeddings, semantic search, prompt or context
engineering, or evaluation of LLM systems. Set it to false if it is off-topic (unrelated
subjects, adverts, cookie banners, or empty text). Give a one-sentence reason."""


# --------------------------------------------------------------------------
# 6. IMAGE DESCRIPTION: turn a diagram/figure into searchable text
# --------------------------------------------------------------------------
IMAGE_PROMPT = """Describe this image for a knowledge base about Agentic RAG.
If it is a diagram or flowchart, list every box/step and what each arrow connects.
If it is a chart or table, give the axes, the main numbers and the conclusion.
Copy any visible text exactly. Be factual and complete; do not guess what is not shown."""

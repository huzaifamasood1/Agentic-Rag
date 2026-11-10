# Research reading file 01: Agentic RAG Survey

**Format:** Original research-oriented study note (not the full text of the cited paper)  
**Topic:** Architecture taxonomy  
**Primary research source:** https://arxiv.org/abs/2501.09136  
**Full paper online:** https://arxiv.org/html/2501.09136 (where available)  
**Publisher PDF:** https://arxiv.org/pdf/2501.09136

## Abstract
This document critically frames a research direction related to agentic retrieval-augmented generation (RAG). It explains the architecture, offers a reproducible experimental design and identifies evaluation criteria and possible open questions. It is independently written for research planning and does not reproduce the original publication.

## Problem statement
Compare fixed retrieval pipelines with agent-controlled loops. Distinguish the agent as a controller from the retriever and generator as tools.

## Technical background and conceptual model
A rigorous survey should distinguish routing, tool use, planning, reflection, multi-agent orchestration and hierarchical control. Each design changes latency, reliability, observability and budget.

A conventional RAG system typically retrieves context with a pre-specified policy and then generates an answer. By contrast, agentic RAG introduces decisions over queries, tools, iterative reading, validation and the stopping condition. These decisions may be rule-based, model-generated or trained. The main scientific challenge is separating the value of better control from extra inference-time computation.

## Proposed experimental methodology
Create a taxonomy matrix covering control flow, source types, number of agents, memory, verification and termination. Build a baseline with a single retrieval call and compare to iterative search with the same source corpus.

### Baselines
- No-retrieval language-model answer.
- Single-pass dense retrieval followed by generation.
- Agentic retrieval with an explicit action budget and recorded tool traces.

### Controlled conditions
Use the same knowledge corpus and generation model across experimental groups. Hold the corpus indexing procedure constant except in experiments that explicitly study indexing. Run multiple seeds for stochastic policies and report confidence intervals. Save the original queries, tool traces, retrieved documents and final citations to permit error analysis.

## Evaluation design
Agentic approaches may improve hard multi-step questions but consume more tokens and tools. Consider retrieval recall, answer grounding, query cost, failures and execution traces.

### Suggested metrics
- **Answer quality:** exact match or task-specific rubric, factual accuracy and completeness.
- **Evidence grounding:** claim support from retrieved passages and unsupported-claim count.
- **Retrieval effectiveness:** evidence recall, ranking quality and proportion of useful retrievals.
- **Efficiency:** latency, tool calls, retrieved tokens and estimated serving cost.
- **Robustness:** performance under ambiguous questions, corrupted evidence and source conflicts.

## Threats to validity
Experiments can overstate benefits when the agent receives more model calls or a larger context window than the baseline. Weak citation judges can confuse plausible statements with evidence-grounded ones. Corpus leakage into model pretraining and cherry-picked complex questions can also distort comparisons. Human review of a stratified sample of answers is therefore recommended.

## Research gap and next question
How can we isolate gains from retrieval policy versus simply spending more compute?

## Source and attribution
This is an **original analysis**, not a conversion or full-text copy of the cited research paper. Read the authoritative paper at https://arxiv.org/abs/2501.09136 to verify the paper's exact claims, methodology and results. The proposed experiment and interpretations here should not be attributed to the original authors.

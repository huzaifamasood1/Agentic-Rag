# Research reading file 07: Agentic RAG and reasoning survey

**Format:** Original research-oriented study note (not the full text of the cited paper)  
**Topic:** Coupling reasoning with retrieval  
**Primary research source:** https://arxiv.org/abs/2507.09477  
**Full paper online:** https://arxiv.org/html/2507.09477 (where available)  
**Publisher PDF:** https://arxiv.org/pdf/2507.09477

## Abstract
This document critically frames a research direction related to agentic retrieval-augmented generation (RAG). It explains the architecture, offers a reproducible experimental design and identifies evaluation criteria and possible open questions. It is independently written for research planning and does not reproduce the original publication.

## Problem statement
Organize methods by whether reasoning improves retrieval, retrieval improves reasoning or both interact repeatedly.

## Technical background and conceptual model
Reasoning-enhanced retrieval can decompose queries, search for missing premises and select tools. Retrieval-enhanced reasoning adds facts to support complex inference. Iterative systems close this loop.

A conventional RAG system typically retrieves context with a pre-specified policy and then generates an answer. By contrast, agentic RAG introduces decisions over queries, tools, iterative reading, validation and the stopping condition. These decisions may be rule-based, model-generated or trained. The main scientific challenge is separating the value of better control from extra inference-time computation.

## Proposed experimental methodology
Categorize each architecture by control policy, evidence format, feedback signal, evaluation dataset and stop criterion.

### Baselines
- No-retrieval language-model answer.
- Single-pass dense retrieval followed by generation.
- Agentic retrieval with an explicit action budget and recorded tool traces.

### Controlled conditions
Use the same knowledge corpus and generation model across experimental groups. Hold the corpus indexing procedure constant except in experiments that explicitly study indexing. Run multiple seeds for stochastic policies and report confidence intervals. Save the original queries, tool traces, retrieved documents and final citations to permit error analysis.

## Evaluation design
Separate factual accuracy, retrieval quality and chain-level faithfulness; compare against no-retrieval and simple-RAG baselines.

### Suggested metrics
- **Answer quality:** exact match or task-specific rubric, factual accuracy and completeness.
- **Evidence grounding:** claim support from retrieved passages and unsupported-claim count.
- **Retrieval effectiveness:** evidence recall, ranking quality and proportion of useful retrievals.
- **Efficiency:** latency, tool calls, retrieved tokens and estimated serving cost.
- **Robustness:** performance under ambiguous questions, corrupted evidence and source conflicts.

## Threats to validity
Experiments can overstate benefits when the agent receives more model calls or a larger context window than the baseline. Weak citation judges can confuse plausible statements with evidence-grounded ones. Corpus leakage into model pretraining and cherry-picked complex questions can also distort comparisons. Human review of a stratified sample of answers is therefore recommended.

## Research gap and next question
How can evaluations tell whether an agent truly uses evidence?

## Source and attribution
This is an **original analysis**, not a conversion or full-text copy of the cited research paper. Read the authoritative paper at https://arxiv.org/abs/2507.09477 to verify the paper's exact claims, methodology and results. The proposed experiment and interpretations here should not be attributed to the original authors.

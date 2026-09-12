# Controlled study

Run `python -m novelty_app.evaluation.run_controlled_study --help` from the repository root.

The new task-first benchmark reuses the four archived nanomedicine snapshots, freezes historical evidence, compares direct/summary/contrastive generation, assesses source support and selects candidates before searching later publications. New LLM calls default to GPT-5.6-Luna with medium reasoning. There is no gold-paper injection or heuristic model fallback.

Commands default to dry runs. Paid stages require explicit execution and per-invocation call/token budgets. Old benchmark outputs and reader ratings are never rewritten.

Secondary publication matching uses a fixed domain-balanced sample of at most 100 tasks, only source-assessor-selected hypotheses, and one joint request for up to ten abstracts per hypothesis. At three repeats this is at most 900 requests before retries. Unassessed hypotheses are not correspondence failures. Primary generation and source assessment still cover the full study.

The complete architecture and execution instructions are maintained with the manuscript:

- [Framework implementation](../../paper/An-Agentic-Framework---nature-communications-ai-computing/FRAMEWORK_IMPROVEMENT_PLAN.md)
- [Experiment execution guide](../../paper/An-Agentic-Framework---nature-communications-ai-computing/EXPERIMENT_REEXECUTION_PLAN.md)

Tests: `python -m pytest tests/test_controlled_study.py -q`.

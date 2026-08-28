# Evidence-Grounded Frontier Mapping and Agentic Hypothesis Generation in Nanomedicine

This code supports the paper, [*Evidence-Grounded Frontier Mapping and Agentic Hypothesis Generation in Nanomedicine*](https://arxiv.org/abs/2605.18144).

Research code for mapping nanotechnology and nanomedicine literature, locating sparse regions in embedding space, and evaluating retrieval-grounded hypothesis-generation workflows.

The project combines a corpus-preparation notebook pipeline, a Streamlit workbench, a snapshot-backed agent service, local embedding/reranking services, and retrospective/prospective evaluation tools.

> **Research-use notice.** This is a research prototype, not a clinical, safety, or experimental-design system. Generated ideas require expert review and independent validation. 

## Contents

- [What is included](#what-is-included)
- [How it works](#how-it-works)
- [Installation and data](#installation-and-data)
- [Run the applications](#run-the-applications)
- [Benchmarking](#benchmarking)
- [Outputs and human review](#outputs-and-human-review)
- [Testing and project notes](#testing-and-project-notes)

## What is included

| Area | Location | Purpose |
| --- | --- | --- |
| Corpus pipeline | `data_setup/` | PubMed collection, cleaning, and embedding-generation notebooks. |
| Novelty workbench | `novelty_app/` | Streamlit UI for loading data, clustering, gap analysis, and snapshot publication. |
| Agent backend | `novelty_app/agents/` | FastAPI service, SQLite knowledge store, evidence-pack retrieval, and LangGraph orchestrator. |
| Benchmark runners | `novelty_app/evaluation/` | Retrospective future-paper recovery and prospective hypothesis generation. |
| Embedding services | `embedding_models/` | Local Qwen embedding/reranking and BioClinical ModernBERT services, plus offline embedding evaluation. |
| Human assessment | `assement_app/` | Blind-first review of generated ideas in an Excel-backed Streamlit app. |
| Tests | `tests/` | Unit and integration-style coverage for the core pipeline. |

Additional subsystem documentation is available in:

- [`embedding_models/README.md`](embedding_models/README.md)
- [`novelty_app/agents/README.md`](novelty_app/agents/README.md)
- [`novelty_app/evaluation/README.md`](novelty_app/evaluation/README.md)
- [`assement_app/README.md`](assement_app/README.md)

## How it works

```text
PubMed records --> cleaning --> document embeddings --> novelty analysis
                                                        |
                                                        v
                                              published analysis snapshot
                                                        |
                                                        v
                            evidence-pack retrieval <-- agent / baseline generators
                                                        |
                                                        v
                         retrospective future-paper recovery or prospective review
```

The novelty-analysis path:

1. loads paper records and aligned precomputed embeddings;
2. optionally reduces dimensionality with PCA;
3. clusters papers with K-means, HDBSCAN, or graph community detection;
4. builds a k-nearest-neighbour graph and multi-`k` density features;
5. combines z-scored density features into a `gap_score` and finds connected high-gap regions;
6. publishes a snapshot that can be retrieved by the backend, generators, and evaluation runners.

Agentic generation uses target-specific evidence packs. The canonical orchestrator performs contrastive explanation, audit, optional patch retrieval, hypothesis generation, scoring, and an experimental-blueprint step. All generated claims should be traced back to papers in the evidence pack; a discovery cue steers retrieval and prompting but is never evidence itself.

## Installation and data

### Requirements

- Python 3.10 or newer
- local corpus and embedding artifacts under `data/`
- a CUDA-capable GPU is strongly recommended for the local Qwen services; CPU execution is possible but slower
- `OPENAI_API_KEY` only for OpenAI-backed methods and LLM UI features

Create and activate an environment from the repository root.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Linux (Bash)**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux distributions that separate venv support, install the `python3-venv` package first. If PowerShell prevents activation, use `Set-ExecutionPolicy -Scope Process RemoteSigned` for the current shell, or activate from `cmd.exe` with `.venv\Scripts\activate.bat`.

### Environment variables

Set secrets in the shell or a secure secret manager—never commit them to source files.

| Variable | When needed | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | LLM methods, LLM analysis, orchestrator | Not required for deterministic baselines. |
| `OPENAI_MODEL` | Optional | Overrides the default model for generation/judging. |
| `NOVELTY_AGENT_DB` | Optional | Overrides `data/novelty_agent_knowledge.sqlite`. |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, `LANGFUSE_TRACING_ENABLED` | Optional | Enables tracing. |
| `QWEN_TORCH_DTYPE` | Optional | CUDA dtype fallback for both Qwen models. |
| `QWEN_EMBED_TORCH_DTYPE`, `QWEN_RERANK_TORCH_DTYPE` | Optional | Per-model CUDA dtype overrides. |
| `QWEN_EMBED_BATCH_SIZE`, `QWEN_RERANK_BATCH_SIZE`, `QWEN_RERANK_MAX_LENGTH`, `QWEN_RERANK_LOGITS_TO_KEEP` | Optional | Lower batch size or maximum length first if reranking runs out of memory. |

For example, set variables for the active shell as follows.

**Windows (PowerShell)**

```powershell
$env:OPENAI_API_KEY = "<your key>"
$env:OPENAI_MODEL = "<model name>"
$env:QWEN_TORCH_DTYPE = "float16"
```

**Linux (Bash)**

```bash
export OPENAI_API_KEY="<your key>"
export OPENAI_MODEL="<model name>"
export QWEN_TORCH_DTYPE="float16"
```

### Data contract

Large data, model, database, and evaluation artifacts are intentionally ignored by Git, so a fresh clone is not self-contained. The usual local inputs are:

```text
data/cleaned_dataset.json
data/all_papers.json
data/qwen_embeddings.npy
data/qwen_embeddings_metadata.json
data/bert_embeddings.npy
data/bert_embeddings_metadata.json
```

For retrospective evaluation, `cleaned_dataset.json` must contain publication-date components (`publication_year`, and preferably `publication_month` and `publication_day`). Each embedding matrix must have exactly one row per dataset record in the same order; the time-split loader rejects misaligned arrays. Keep the record order stable throughout extraction, preprocessing, and embedding creation.

Use the notebooks in `data_setup/` to build or reproduce these artifacts:

1. `1.pubmed_data_extraction.ipynb`
2. `2.data_preprocessing.ipynb`
3. `3.create_bert_embeddings.ipynb` or `3.create_qwen_embeddings.ipynb`

## Run the applications

### Novelty workbench

```console
streamlit run novelty_app/app.py
```

The workbench covers data/configuration, embedding processing, filtering, clustering, gap analysis, published snapshots, agent controls, database exploration, and export.

### Qwen embedding and reranking service

Start this in a separate shell. Run the command matching your shell:

**Windows (PowerShell)**

```powershell
Set-Location embedding_models
uvicorn qwen:app --host 0.0.0.0 --port 8000
```

**Linux (Bash)**

```bash
cd embedding_models
uvicorn qwen:app --host 0.0.0.0 --port 8000
```

The service supplies both embeddings and cross-encoder reranking and is the retrieval backend used by the evaluation stack. Use `http://127.0.0.1:8000` from local CLI clients. See its interactive API documentation at `http://127.0.0.1:8000/docs`.

The BioClinical ModernBERT service can be run independently on port 8001:

**Windows (PowerShell)**

```powershell
Set-Location embedding_models
uvicorn bert:app --host 0.0.0.0 --port 8001
```

**Linux (Bash)**

```bash
cd embedding_models
uvicorn bert:app --host 0.0.0.0 --port 8001
```

### Snapshot and agent backend

Start this in another shell from the repository root:

```console
uvicorn agents.backend_api:app --app-dir novelty_app --host 0.0.0.0 --port 8088
```

The backend persists published snapshots, papers, clusters, gap membership, evidence packs, generated artifacts, and evaluation runs. Its default SQLite database is `data/novelty_agent_knowledge.sqlite`.

### Interactive agent run

With the backend running:

```console
python -m novelty_app.agents.run_interactive
```

This selects a published snapshot and a gap or cluster-pair target, then runs the orchestrator outside Streamlit.

## Benchmarking

### What the retrospective benchmark measures

The main benchmark asks: *using evidence available before a time cutoff, can a method generate a hypothesis whose retrieval results recover a later, target-associated paper?*

For each run, the evaluator:

1. splits the corpus into historical and future windows;
2. builds or reuses a **historical** analysis snapshot;
3. selects historical gap and cluster-pair targets;
4. constructs focused evidence packs and assigns eligible future papers to targets;
5. generates ideas with each configured method and seed;
6. retrieves over the held-out future corpus and separately checks historical confounders;
7. stores raw matches, aggregate metrics, and review material.

The standard methods are:

| Method | Role |
| --- | --- |
| `orchestrator` | Multi-step evidence-grounded LangGraph workflow. |
| `single_shot_llm` | Single-pass LLM generator. |
| `cue_retrieval_generation` | Cue-only retrieve-then-generate LLM baseline; receives no frontier structure. |
| `retrieval_summary_direct` | LLM generator based directly on retrieval summaries. |
| `heuristic_bridge` | Deterministic bridge-style baseline. |
| `pack_query_baseline` | Deterministic retrieval-oriented baseline derived from the evidence pack. |
| `random_target_control` | Control that breaks the intended target relationship. |

`orchestrator`, `single_shot_llm`, `cue_retrieval_generation`, and `retrieval_summary_direct` require `OPENAI_API_KEY`. The deterministic methods make a useful service and data smoke test without an API key. Because `cue_retrieval_generation` is defined by cue-only retrieval, select it only when a discovery cue and cue-source snapshot are configured.

### Reproducible smoke run

Before running the command, start the Qwen service and the agent backend as described above. From the repository root, run a small, deterministic comparison.

**Windows (PowerShell)**

```powershell
python -m novelty_app.evaluation.run_retrospective `
  --backend-url http://127.0.0.1:8088 `
  --qwen-base-url http://127.0.0.1:8000 `
  --data-json data/cleaned_dataset.json `
  --data-dir data `
  --n-gap-targets 1 `
  --n-cluster-pair-targets 1 `
  --n-gold-future-papers 2 `
  --methods heuristic_bridge pack_query_baseline random_target_control `
  --seeds 1 `
  --hypotheses-per-target 1 `
  --analysis-clustering-method kmeans `
  --analysis-pca-components 16 `
  --output-dir data/retrospective_eval_smoke
```

**Linux (Bash)**

```bash
python -m novelty_app.evaluation.run_retrospective \
  --backend-url http://127.0.0.1:8088 \
  --qwen-base-url http://127.0.0.1:8000 \
  --data-json data/cleaned_dataset.json \
  --data-dir data \
  --n-gap-targets 1 \
  --n-cluster-pair-targets 1 \
  --n-gold-future-papers 2 \
  --methods heuristic_bridge pack_query_baseline random_target_control \
  --seeds 1 \
  --hypotheses-per-target 1 \
  --analysis-clustering-method kmeans \
  --analysis-pca-components 16 \
  --output-dir data/retrospective_eval_smoke
```

Use a separate output directory for every run. The default benchmark uses a cutoff of `2020-12-31` and a future window of `2022-01-01` through `2025-12-31`; record any changes to those values alongside results.

The remaining multiline benchmark examples use PowerShell syntax. For Linux, replace each trailing backtick (`` ` ``) with a backslash (`\`); the arguments and relative paths are otherwise identical. Alternatively, place the command on one line.

To compare LLM and non-LLM methods, set `OPENAI_API_KEY` and replace `--methods` with the complete method list shown above. For a paper-grade run, increase targets, future papers, seeds, and hypotheses per target only after inspecting a smoke-run review packet.

### Cues, cached snapshots, and leakage controls

A discovery cue can steer selection, retrieval, and prompting. The current CLI requires `--cue-source-snapshot-id` whenever a cue is active; it identifies the published snapshot used for cue-semantic retrieval. For example:

```powershell
python -m novelty_app.evaluation.run_retrospective `
  --backend-url http://127.0.0.1:8088 `
  --qwen-base-url http://127.0.0.1:8000 `
  --existing-snapshot-id <historical_snapshot_id> `
  --cue-source-snapshot-id <historical_snapshot_id> `
  --discovery-cue-text "Focus on folate-targeted RNA delivery for breast cancer" `
  --n-gap-targets 2 `
  --n-cluster-pair-targets 2 `
  --n-gold-future-papers 10 `
  --methods orchestrator heuristic_bridge pack_query_baseline random_target_control `
  --seeds 1 `
  --hypotheses-per-target 1 `
  --output-dir data/retrospective_eval_cued
```

The cue is not literature evidence and should not be cited as support. It reranks eligible future papers rather than hard-filtering them, and cue-aware metrics are reported when one is supplied.

`cue_retrieval_generation` provides the conventional retrieval-first comparison. It retrieves historical evidence using only the normalized discovery cue (the top-ranked semantic cue results plus the cue's compiled lexical queries) and passes those papers directly to the structured LLM generator. The assigned gap or cluster pair is retained only as the evaluation label and is not sent to retrieval, retrieval seeding, or generation. This differs from `retrieval_summary_direct`, which retrieves the assigned frontier evidence pack and then summarizes it before generation.

To run this baseline over the archived four-domain nanomedicine benchmark and place each result alongside the existing methods under `data/nanomedicine/baseline_runs/<domain>/cue_retrieval_generation`, use:

```powershell
python -m novelty_app.evaluation.run_nanomedicine_baselines `
  --qwen-base-url http://192.168.2.35:8000 `
  --methods cue_retrieval_generation
```

For valid comparisons:

- reuse the same corpus version, date split, snapshot configuration, target counts, seed set, and retrieval service across methods;
- only pass an `--existing-snapshot-id` known to contain historical papers for the requested cutoff;
- preserve the default historical near-duplicate screen. `--disable-leakage-check` is a debugging/speed option and should not be used for reported results without explicit justification;
- inspect the exported review packet for target assignment, future neighbours, and historical confounders before drawing conclusions;
- report failures and incomplete tasks as well as aggregate scores.

### Metrics and interpretation

Primary recovery labels are `gold_recovered`, `future_neighbor_only`, `historical_confound`, and `not_recovered`. Aggregate output includes `gold_recall_at_1`, `gold_recall_at_5`, `gold_recall_at_10`, `gold_mrr`, `gold_recovered_rate`, `historical_confound_rate`, `median_gold_rank`, and idea-quality summaries. Cue-aware runs additionally report cue-weighted recall/MRR and hypothesis cue alignment.

Higher recovery does not by itself establish that an idea is useful or novel: a future paper may be semantically related but scientifically uninteresting, while a valuable idea might not have a matching future paper. Use blind expert assessment alongside the automatic metrics.

### Prospective generation

Prospective generation runs the same generator registry against an existing snapshot without time splitting or future-paper recovery. It is intended for producing candidates for review, not for measuring benchmark performance.

```powershell
python -m novelty_app.evaluation.run_prospective `
  --backend-url http://127.0.0.1:8088 `
  --snapshot-id <snapshot_id> `
  --n-gap-targets 2 `
  --n-cluster-pair-targets 2 `
  --methods orchestrator heuristic_bridge pack_query_baseline `
  --seeds 1 `
  --hypotheses-per-target 1 `
  --output-dir data/prospective_eval_smoke
```

Use `--gap-id <id>` or `--cluster-pair <cluster_a> <cluster_b>` to target a specific frontier. Discovery cues again require `--cue-source-snapshot-id`.

### Offline embedding benchmark

`embedding_models/eval.py` checks a precomputed embedding matrix independently of the agent workflow. It runs repeated-split multilabel linear probing, keyword-overlap retrieval, and an optional TF-IDF baseline. It rejects row-count mismatches by default.

```powershell
python embedding_models/eval.py `
  --data_json data/cleaned_dataset.json `
  --embeddings_npy data/qwen_embeddings.npy `
  --keyword_col mesh `
  --min_keyword_freq 5 `
  --probe_backend auto `
  --threshold_tuning auto `
  --probe_n_jobs 1 `
  --n_repeats 3 `
  --test_size 0.2 `
  --base_seed 42 `
  --k_retrieval 10 `
  --max_retrieval_queries 5000 `
  --progress `
  --output_json data/qwen_evaluation_report.json
```

The report includes probe F1/precision/recall, LRAP, ranking loss, MRR, precision/recall/hit rate/MAP/nDCG at `k`, query coverage, and the retained keyword classes. These metrics assess representation quality under keyword-overlap relevance; they do not substitute for the frontier-recovery benchmark.

## Outputs and human review

The retrospective runner stores run and match records in the backend database and writes the following to `--output-dir`:

- `<run_id>_review_packet.csv` and `<run_id>_review_packet.json`: best hypothesis per benchmark task, target, evidence summary, retrievals, cue data, and blank review fields;
- `<run_id>_assessment_bundle_v1.json`: the complete blind-review payload.

The prospective runner writes `<run_id>_summary.json`, `<run_id>_hypotheses.json`, and `<run_id>_hypotheses.csv`.

Open the human review interface with:

```powershell
streamlit run assement_app/app.py
```

Choose **Retrospective** to load an `assessment_bundle_v1` file or **Prospective** to load a hypotheses JSON file. The app writes reviewer state to an Excel workbook with `meta`, `ideas`, `assessments`, and `summary` sheets.

## Testing and project notes

Run the automated test suite from the repository root:

```powershell
pytest
```

Useful checks before a large run:

1. confirm the Qwen and backend health endpoints are reachable;
2. run the deterministic smoke benchmark;
3. inspect the review packet and SQLite run record;
4. only then scale the method, seed, target, and future-paper counts.

Notes:

- `assement_app/` is intentionally spelled this way because it is the current import and run path.
- `paper/` contains manuscript and figure assets rather than runtime code.
- Model weights may be downloaded by the local services on first use; ensure the host has appropriate Hugging Face access and storage.
- No citation metadata or licence file is currently provided in the repository. Add an explicit citation and licence before external distribution.

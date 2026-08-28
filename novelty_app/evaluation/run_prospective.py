from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO, Tuple

try:
    from agents.backend_client import BackendClient
except Exception:  # pragma: no cover
    from novelty_app.agents.backend_client import BackendClient

from novelty_app.agents.observability import (
    current_trace_ref,
    deterministic_trace_id,
    flush_langfuse,
    langfuse_status,
    observe_current,
    trace_attributes,
)
from novelty_app.discovery_cue import discovery_cue_to_dict

from .generators import GENERATOR_REGISTRY, GenerationContext, run_generation_method, target_id
from .judge import score_hypotheses


DEFAULT_METHODS = [
    "orchestrator",
    "single_shot_llm",
    "retrieval_summary_direct",
    "heuristic_bridge",
    "pack_query_baseline",
    "random_target_control",
]

_LLM_REQUIRED_METHODS = {
    "orchestrator",
    "single_shot_llm",
    "cue_retrieval_generation",
    "retrieval_summary_direct",
}


@dataclass
class ProspectiveResult:
    run: Dict[str, Any]
    tasks: List[Dict[str, Any]]
    failures: List[Dict[str, Any]]
    summary_json: str
    hypotheses_json: str
    hypotheses_csv: str


@dataclass
class ProspectiveProgress:
    phase: str
    status: str = "running"
    current: Optional[int] = None
    total: Optional[int] = None
    message: str = ""
    run_id: str = ""
    method_name: Optional[str] = None
    seed: Optional[int] = None
    target_id: Optional[str] = None
    n_tasks: int = 0
    n_failures: int = 0

    def to_payload(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class _ProspectiveCliProgressReporter:
    def __init__(self, stream: Optional[TextIO] = None) -> None:
        self._stream = stream or sys.stdout

    def __call__(self, progress: ProspectiveProgress) -> None:
        parts = [f"[{progress.phase}]"]
        if progress.message:
            parts.append(progress.message)
        if progress.current is not None and progress.total is not None:
            parts.append(f"({progress.current}/{progress.total})")
        if progress.phase in {"generating_tasks", "completed", "failed"}:
            parts.append(f"tasks={progress.n_tasks}")
            parts.append(f"failures={progress.n_failures}")
        self._stream.write(" ".join(parts) + "\n")
        self._stream.flush()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _sanitize_id_fragment(value: str, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value or ""))
    cleaned = cleaned.strip("_")
    return cleaned[:48] or fallback


def _normalize_required_paper_ids(values: Optional[Sequence[Any]]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _target_summary(target: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"target_type": target.get("target_type")}
    if target.get("target_type") == "gap":
        summary["gap_id"] = target.get("gap_id")
    else:
        summary["cluster_a"] = target.get("cluster_a")
        summary["cluster_b"] = target.get("cluster_b")
    return {key: value for key, value in summary.items() if value is not None}


def _safe_target_id(target: Dict[str, Any]) -> str:
    try:
        return target_id(target)
    except Exception:
        return _json_dumps(_target_summary(target))


def _summarize_evidence_pack(pack: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = dict(pack or {})
    papers = list(payload.get("papers") or [])
    compact_papers: List[Dict[str, Any]] = []
    for paper in papers[:5]:
        compact_papers.append(
            {
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title", ""),
                "year": paper.get("publication_year", paper.get("year")),
                "selection_sources": list(paper.get("selection_sources") or []),
            }
        )
    return {
        "stats": dict(payload.get("stats") or {}),
        "meta": dict(payload.get("meta") or {}),
        "top_papers": compact_papers,
        "n_papers": len(papers),
    }


def _select_cluster_pair_targets(
    backend: BackendClient,
    snapshot_id: str,
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []

    pairs: List[Dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    top_gap_resp = backend.top_gaps(snapshot_id=snapshot_id, k=max(50, limit * 4))
    for gap in top_gap_resp.get("gaps", []):
        cluster_ids = sorted(
            {
                int(cluster_id)
                for cluster_id in (gap.get("cluster_ids") or [])
                if cluster_id is not None and str(cluster_id) not in {"", "-1"}
            }
        )
        for cluster_a, cluster_b in combinations(cluster_ids, 2):
            pair = (cluster_a, cluster_b)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(
                {
                    "target_type": "cluster_pair",
                    "cluster_a": cluster_a,
                    "cluster_b": cluster_b,
                    "source_gap_id": gap.get("gap_id"),
                }
            )
            if len(pairs) >= limit:
                return pairs

    clusters = backend.list_clusters(snapshot_id=snapshot_id, limit=max(20, limit * 3)).get("clusters", [])
    ordered = [int(cluster["cluster_id"]) for cluster in clusters]
    for cluster_a, cluster_b in combinations(ordered, 2):
        pair = tuple(sorted((cluster_a, cluster_b)))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append({"target_type": "cluster_pair", "cluster_a": pair[0], "cluster_b": pair[1]})
        if len(pairs) >= limit:
            break
    return pairs


def _explicit_targets(
    *,
    gap_ids: Optional[Sequence[str]] = None,
    cluster_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for gap_id in gap_ids or []:
        text = str(gap_id or "").strip()
        if text:
            targets.append({"target_type": "gap", "gap_id": text})
    for cluster_a, cluster_b in cluster_pairs or []:
        targets.append(
            {
                "target_type": "cluster_pair",
                "cluster_a": int(cluster_a),
                "cluster_b": int(cluster_b),
            }
        )
    return targets


def _dedupe_targets(targets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for target in targets:
        key = _safe_target_id(target)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(target))
    return out


def _resolve_targets(
    backend: BackendClient,
    snapshot_id: str,
    *,
    explicit_targets: Optional[Sequence[Dict[str, Any]]] = None,
    n_gap_targets: int,
    n_cluster_pair_targets: int,
) -> List[Dict[str, Any]]:
    if explicit_targets:
        return _dedupe_targets([dict(target) for target in explicit_targets])

    gap_targets = backend.top_gaps(snapshot_id=snapshot_id, k=max(0, n_gap_targets)).get("gaps", [])
    gap_target_rows = [
        {"target_type": "gap", "gap_id": gap["gap_id"], "source_rank": idx}
        for idx, gap in enumerate(gap_targets[: max(0, n_gap_targets)])
    ]
    cluster_targets = _select_cluster_pair_targets(
        backend,
        snapshot_id,
        limit=max(0, n_cluster_pair_targets),
    )
    return _dedupe_targets([*gap_target_rows, *cluster_targets])


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return round(sum(filtered) / float(len(filtered)), 4)


def _build_run_summary(tasks: Sequence[Dict[str, Any]], failures: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    method_summary: Dict[str, Dict[str, Any]] = {}
    all_scores: List[Optional[float]] = []
    published_count = 0
    hypothesis_count = 0

    for task in tasks:
        method_name = str(task.get("method_name") or "")
        bucket = method_summary.setdefault(
            method_name,
            {
                "n_tasks": 0,
                "n_hypotheses": 0,
                "n_published": 0,
                "_scores": [],
            },
        )
        bucket["n_tasks"] += 1
        if task.get("published"):
            bucket["n_published"] += 1
            published_count += 1

        for hypothesis in task.get("hypotheses", []):
            score = (hypothesis.get("idea_scores") or {}).get("average_score")
            numeric_score = float(score) if score is not None else None
            all_scores.append(numeric_score)
            if numeric_score is not None:
                bucket["_scores"].append(numeric_score)
            hypothesis_count += 1
        bucket["n_hypotheses"] += len(task.get("hypotheses", []))

    for bucket in method_summary.values():
        bucket["mean_average_idea_score"] = _mean(bucket.pop("_scores", []))

    return {
        "n_completed_tasks": len(tasks),
        "n_failed_tasks": len(failures),
        "n_total_tasks": len(tasks) + len(failures),
        "n_generated_hypotheses": hypothesis_count,
        "n_published_tasks": published_count,
        "mean_average_idea_score": _mean(all_scores),
        "methods": method_summary,
    }


def _flatten_hypothesis_rows(run_payload: Dict[str, Any], tasks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        target = dict(task.get("target") or {})
        effective_target = dict(task.get("effective_target") or target)
        effective_target_summary = _target_summary(effective_target)
        evidence_pack = dict(task.get("evidence_pack") or {})
        evidence_paper_ids = [
            paper.get("paper_id")
            for paper in list(evidence_pack.get("papers") or [])[:10]
            if paper.get("paper_id")
        ]
        published_artifact = dict(task.get("published_artifact") or {})
        task_trace_ref = dict(task.get("trace_ref") or {})
        for hypothesis in task.get("hypotheses", []):
            idea_scores = dict(hypothesis.get("idea_scores") or {})
            hypothesis_trace_ref = dict(hypothesis.get("trace_ref") or {})
            rows.append(
                {
                    "run_id": run_payload["run_id"],
                    "snapshot_id": run_payload["snapshot_id"],
                    "method_name": task.get("method_name"),
                    "seed": task.get("seed"),
                    "target_id": task.get("target_id"),
                    "target_type": task.get("target_type"),
                    "gap_id": target.get("gap_id"),
                    "cluster_a": target.get("cluster_a"),
                    "cluster_b": target.get("cluster_b"),
                    "effective_target_id": task.get("effective_target_id"),
                    "effective_target_type": effective_target_summary.get("target_type"),
                    "effective_gap_id": effective_target_summary.get("gap_id"),
                    "effective_cluster_a": effective_target_summary.get("cluster_a"),
                    "effective_cluster_b": effective_target_summary.get("cluster_b"),
                    "hypothesis_id": hypothesis.get("hypothesis_id"),
                    "title": hypothesis.get("title"),
                    "text": hypothesis.get("text"),
                    "average_score": idea_scores.get("average_score"),
                    "score_method": idea_scores.get("score_method"),
                    "judge_model": idea_scores.get("judge_model"),
                    "support_citations": _json_dumps(hypothesis.get("support_citations") or []),
                    "idea_scores": _json_dumps(idea_scores),
                    "evidence_size": len(evidence_pack.get("papers") or []),
                    "evidence_paper_ids": _json_dumps(evidence_paper_ids),
                    "published": bool(task.get("published")),
                    "published_artifact_id": published_artifact.get("artifact_id"),
                    "task_trace_id": task_trace_ref.get("trace_id"),
                    "task_trace_url": task_trace_ref.get("url"),
                    "hypothesis_trace_id": hypothesis_trace_ref.get("trace_id"),
                    "hypothesis_trace_url": hypothesis_trace_ref.get("url"),
                }
            )
    return rows


def _write_outputs(
    output_dir: Path,
    *,
    run_payload: Dict[str, Any],
    tasks: Sequence[Dict[str, Any]],
    failures: Sequence[Dict[str, Any]],
) -> Tuple[str, str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(run_payload["run_id"])
    summary_path = (output_dir / f"{run_id}_summary.json").resolve()
    hypotheses_json_path = (output_dir / f"{run_id}_hypotheses.json").resolve()
    hypotheses_csv_path = (output_dir / f"{run_id}_hypotheses.csv").resolve()

    summary_payload = {
        "run": run_payload,
        "failures": list(failures),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    hypotheses_payload = {
        "run": run_payload,
        "tasks": list(tasks),
        "failures": list(failures),
    }
    hypotheses_json_path.write_text(json.dumps(hypotheses_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = _flatten_hypothesis_rows(run_payload, tasks)
    fieldnames = [
        "run_id",
        "snapshot_id",
        "method_name",
        "seed",
        "target_id",
        "target_type",
        "gap_id",
        "cluster_a",
        "cluster_b",
        "effective_target_id",
        "effective_target_type",
        "effective_gap_id",
        "effective_cluster_a",
        "effective_cluster_b",
        "hypothesis_id",
        "title",
        "text",
        "average_score",
        "score_method",
        "judge_model",
        "support_citations",
        "idea_scores",
        "evidence_size",
        "evidence_paper_ids",
        "published",
        "published_artifact_id",
        "task_trace_id",
        "task_trace_url",
        "hypothesis_trace_id",
        "hypothesis_trace_url",
    ]
    with hypotheses_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return str(summary_path), str(hypotheses_json_path), str(hypotheses_csv_path)


def run_prospective(
    *,
    snapshot_id: str,
    backend_url: str = "http://localhost:8088",
    methods: Sequence[str] = DEFAULT_METHODS,
    seeds: int = 3,
    hypotheses_per_target: int = 3,
    n_gap_targets: int = 20,
    n_cluster_pair_targets: int = 10,
    explicit_targets: Optional[Sequence[Dict[str, Any]]] = None,
    output_dir: str = "data/prospective_eval",
    openai_api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    discovery_cue: Optional[Any] = None,
    cue_source_snapshot_id: Optional[str] = None,
    cue_similarity_top_k: int = 50,
    cue_similarity_sample_n: int = 6,
    exemplars: int = 8,
    boundary: int = 8,
    diverse: int = 0,
    max_iters: int = 2,
    required_paper_ids: Optional[Sequence[str]] = None,
    required_paper_source_snapshot_id: Optional[str] = None,
    progress_callback: Optional[Callable[[ProspectiveProgress], None]] = None,
) -> ProspectiveResult:
    snapshot_id = str(snapshot_id or "").strip()
    if not snapshot_id:
        raise ValueError("snapshot_id is required")
    if seeds <= 0:
        raise ValueError("seeds must be >= 1")
    if hypotheses_per_target <= 0:
        raise ValueError("hypotheses_per_target must be >= 1")

    unknown_methods = [method for method in methods if method not in GENERATOR_REGISTRY]
    if unknown_methods:
        raise ValueError(f"Unknown generation methods: {unknown_methods}")

    normalized_methods = list(dict.fromkeys(str(method) for method in methods))
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if any(method in _LLM_REQUIRED_METHODS for method in normalized_methods) and not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for methods: "
            + ", ".join(method for method in normalized_methods if method in _LLM_REQUIRED_METHODS)
        )

    backend = BackendClient(backend_url)
    snapshot = backend.get_snapshot(snapshot_id)
    resolved_snapshot_id = str(snapshot.get("snapshot_id") or snapshot_id)
    normalized_required_paper_source_snapshot_id = str(required_paper_source_snapshot_id or "").strip() or None
    resolved_required_paper_source_snapshot_id = resolved_snapshot_id
    if normalized_required_paper_source_snapshot_id:
        if normalized_required_paper_source_snapshot_id == resolved_snapshot_id:
            resolved_required_paper_source_snapshot_id = resolved_snapshot_id
        else:
            required_source_snapshot = backend.get_snapshot(normalized_required_paper_source_snapshot_id)
            resolved_required_paper_source_snapshot_id = str(
                required_source_snapshot.get("snapshot_id") or normalized_required_paper_source_snapshot_id
            )
    normalized_required_paper_ids = _normalize_required_paper_ids(required_paper_ids)
    if normalized_required_paper_ids:
        batch_payload = backend.papers_batch(
            snapshot_id=resolved_required_paper_source_snapshot_id,
            paper_ids=normalized_required_paper_ids,
            fields=["paper_id"],
        )
        unresolved_ids = [
            str(paper_id).strip()
            for paper_id in (batch_payload.get("unresolved_paper_ids") or [])
            if str(paper_id).strip()
        ]
        if unresolved_ids:
            raise ValueError(
                "required_paper_ids not found in source snapshot "
                f"`{resolved_required_paper_source_snapshot_id}`: {', '.join(unresolved_ids[:10])}"
            )
        ambiguous_ids = batch_payload.get("ambiguous_paper_ids") or {}
        if ambiguous_ids:
            alias, candidates = next(iter(ambiguous_ids.items()))
            raise ValueError(
                "required_paper_ids are ambiguous in source snapshot "
                f"`{resolved_required_paper_source_snapshot_id}` for `{alias}`: {', '.join(candidates[:10])}"
            )
        resolved_required_ids = [
            str(paper_id).strip()
            for paper_id in (batch_payload.get("resolved_paper_ids") or [])
            if str(paper_id).strip()
        ]
        if resolved_required_ids:
            normalized_required_paper_ids = resolved_required_ids
        else:
            fetched_papers = batch_payload.get("papers", [])
            found_ids = {str(paper.get("paper_id") or "").strip() for paper in fetched_papers if paper.get("paper_id")}
            missing_ids = [paper_id for paper_id in normalized_required_paper_ids if paper_id not in found_ids]
            if missing_ids:
                raise ValueError(
                    "required_paper_ids not found in source snapshot "
                    f"`{resolved_required_paper_source_snapshot_id}`: {', '.join(missing_ids[:10])}"
                )
    normalized_cue = discovery_cue_to_dict(discovery_cue)
    cue_source_snapshot_id = str(cue_source_snapshot_id or "").strip() or None
    if normalized_cue and not cue_source_snapshot_id:
        raise ValueError("cue_source_snapshot_id is required when discovery_cue is active.")
    cue_similarity_top_k = int(cue_similarity_top_k)
    cue_similarity_sample_n = int(cue_similarity_sample_n)
    if cue_similarity_top_k < 1:
        raise ValueError("cue_similarity_top_k must be >= 1.")
    if cue_similarity_sample_n < 0:
        raise ValueError("cue_similarity_sample_n must be >= 0.")
    targets = _resolve_targets(
        backend,
        resolved_snapshot_id,
        explicit_targets=explicit_targets,
        n_gap_targets=n_gap_targets,
        n_cluster_pair_targets=n_cluster_pair_targets,
    )
    if not targets:
        raise ValueError("No prospective targets were selected. Provide explicit targets or increase target counts.")

    run_id = f"prospective_{_sanitize_id_fragment(resolved_snapshot_id, 'snapshot')}_{uuid.uuid4().hex[:8]}"
    run_payload: Dict[str, Any] = {
        "run_id": run_id,
        "snapshot_id": resolved_snapshot_id,
        "created_at": _utc_now_iso(),
        "method_names": normalized_methods,
        "config": {
            "backend_url": backend_url,
            "seeds": seeds,
            "hypotheses_per_target": hypotheses_per_target,
            "n_gap_targets": n_gap_targets,
            "n_cluster_pair_targets": n_cluster_pair_targets,
            "explicit_targets": [dict(target) for target in explicit_targets or []],
            "exemplars": exemplars,
            "boundary": boundary,
            "diverse": diverse,
            "max_iters": max_iters,
            "required_paper_ids": list(normalized_required_paper_ids),
            "required_paper_source_snapshot_id": (
                resolved_required_paper_source_snapshot_id
                if normalized_required_paper_source_snapshot_id
                else None
            ),
            "model_name": model_name or os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07"),
            "cue_source_snapshot_id": cue_source_snapshot_id,
            "cue_similarity_top_k": cue_similarity_top_k,
            "cue_similarity_sample_n": cue_similarity_sample_n,
        },
        "status": "running",
        "summary": {},
        "discovery_cue": dict(normalized_cue or {}),
        "snapshot_metadata": dict(snapshot.get("metadata") or {}),
        "langfuse": langfuse_status(),
        "observability": {},
        "output_files": {},
    }

    tasks: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    total_tasks = len(normalized_methods) * seeds * len(targets)

    def _emit_progress(
        phase: str,
        *,
        status: str = "running",
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: str = "",
        method_name: Optional[str] = None,
        seed: Optional[int] = None,
        target_label: Optional[str] = None,
    ) -> None:
        if progress_callback is None:
            return
        progress = ProspectiveProgress(
            phase=phase,
            status=status,
            current=current,
            total=total,
            message=message,
            run_id=run_id,
            method_name=method_name,
            seed=seed,
            target_id=target_label,
            n_tasks=len(tasks),
            n_failures=len(failures),
        )
        progress_callback(progress)

    try:
        _emit_progress("validating_snapshot", message=f"Using snapshot `{resolved_snapshot_id}`")
        _emit_progress(
            "selecting_targets",
            current=len(targets),
            total=len(targets),
            message=f"Selected {len(targets)} prospective targets",
        )

        run_trace_id = deterministic_trace_id(run_id)
        run_tags = ["prospective_run", "snapshot"]
        if normalized_cue:
            run_tags.append("discovery_cue")

        with observe_current(
            name="prospective_run",
            as_type="agent",
            input_payload={
                "snapshot_id": resolved_snapshot_id,
                "method_names": normalized_methods,
                "n_targets": len(targets),
                "seeds": seeds,
            },
            metadata={
                "run_id": run_id,
                "snapshot_id": resolved_snapshot_id,
                "n_targets": len(targets),
                "methods": ",".join(normalized_methods),
            },
            trace_id=run_trace_id,
        ) as run_observation:
            with trace_attributes(
                session_id=run_id,
                tags=run_tags,
                trace_name="prospective_run",
                metadata={
                    "run_id": run_id,
                    "snapshot_id": resolved_snapshot_id,
                    "n_targets": len(targets),
                },
            ):
                run_payload["observability"] = current_trace_ref(
                    session_id=run_id,
                    tags=run_tags,
                    metadata={
                        "run_id": run_id,
                        "snapshot_id": resolved_snapshot_id,
                    },
                )

                task_counter = 0
                _emit_progress(
                    "generating_tasks",
                    current=0,
                    total=total_tasks,
                    message="Running prospective generation tasks" if total_tasks > 0 else "No tasks to run",
                )

                for method_name in normalized_methods:
                    for seed in range(seeds):
                        for target in targets:
                            target_label = _safe_target_id(target)
                            task_metadata = {
                                "run_id": run_id,
                                "snapshot_id": resolved_snapshot_id,
                                "method_name": method_name,
                                "seed": seed,
                                **_target_summary(target),
                            }
                            task_tags = ["prospective_run", "generation_task", method_name]
                            target_type = str(target.get("target_type") or "")
                            if target_type:
                                task_tags.append(target_type)
                            task_trace_ref: Dict[str, Any] = {}
                            try:
                                with observe_current(
                                    name="prospective_generation_task",
                                    as_type="agent",
                                    input_payload=task_metadata,
                                    metadata=task_metadata,
                                    trace_id=deterministic_trace_id(f"{run_id}:{method_name}:{seed}:{target_label}"),
                                ) as task_observation:
                                    with trace_attributes(
                                        session_id=run_id,
                                        tags=task_tags,
                                        trace_name="prospective_generation_task",
                                        metadata=task_metadata,
                                    ):
                                        task_trace_ref = current_trace_ref(
                                            session_id=run_id,
                                            tags=task_tags,
                                            metadata=task_metadata,
                                        )
                                        context = GenerationContext(
                                            backend=backend,
                                            snapshot_id=resolved_snapshot_id,
                                            target=dict(target),
                                            seed=seed,
                                            openai_api_key=openai_api_key,
                                            model_name=model_name,
                                            discovery_cue=normalized_cue,
                                            cue_source_snapshot_id=cue_source_snapshot_id,
                                            cue_similarity_top_k=cue_similarity_top_k,
                                            cue_similarity_sample_n=cue_similarity_sample_n,
                                            cue_similarity_seed=f"{run_id}:{method_name}:{seed}:{target_label}",
                                            hypotheses_per_target=hypotheses_per_target,
                                            exemplars=exemplars,
                                            boundary=boundary,
                                            diverse=diverse,
                                            max_iters=max_iters,
                                            required_paper_ids=normalized_required_paper_ids,
                                            required_paper_source_snapshot_id=(
                                                resolved_required_paper_source_snapshot_id
                                                if normalized_required_paper_source_snapshot_id
                                                else None
                                            ),
                                        )
                                        generated_hypotheses, gen_meta = run_generation_method(method_name, context)
                                        effective_target = dict(gen_meta.get("effective_target") or target)
                                        evidence_pack = dict(gen_meta.get("evidence_pack") or {})
                                        scored_hypotheses = score_hypotheses(
                                            [hypothesis.model_dump() for hypothesis in generated_hypotheses],
                                            evidence_pack=evidence_pack,
                                            audit=gen_meta.get("audit"),
                                            explanation=gen_meta.get("explanation"),
                                            target=effective_target,
                                            discovery_cue=normalized_cue,
                                            openai_api_key=openai_api_key,
                                            model_name=model_name,
                                        )

                                        hypothesis_rows: List[Dict[str, Any]] = []
                                        for hypothesis in generated_hypotheses:
                                            hypothesis.run_id = run_id
                                            hypothesis.idea_scores = dict(
                                                scored_hypotheses.get(hypothesis.hypothesis_id) or {}
                                            )
                                            hyp_payload = hypothesis.model_dump()
                                            if not hyp_payload.get("trace_ref") and task_trace_ref:
                                                hyp_payload["trace_ref"] = dict(task_trace_ref)
                                            hypothesis_rows.append(hyp_payload)

                                        task_row = {
                                            "task_id": f"{method_name}:{seed}:{target_label}",
                                            "run_id": run_id,
                                            "snapshot_id": resolved_snapshot_id,
                                            "method_name": method_name,
                                            "seed": seed,
                                            "target": dict(target),
                                            "target_id": target_label,
                                            "target_type": str(target.get("target_type") or ""),
                                            "effective_target": effective_target,
                                            "effective_target_id": _safe_target_id(effective_target),
                                            "hypotheses": hypothesis_rows,
                                            "n_hypotheses": len(hypothesis_rows),
                                            "evidence_pack": evidence_pack,
                                            "evidence_pack_summary": _summarize_evidence_pack(evidence_pack),
                                            "explanation": dict(gen_meta.get("explanation") or {}),
                                            "audit": dict(gen_meta.get("audit") or {}),
                                            "blueprint": dict(gen_meta.get("blueprint") or {}),
                                            "published": bool(gen_meta.get("published")),
                                            "published_artifact": dict(gen_meta.get("published_artifact") or {}),
                                            "observability": dict(gen_meta.get("observability") or {}),
                                            "iterations": int(gen_meta.get("iterations") or 0),
                                            "trace_ref": dict(task_trace_ref),
                                        }
                                        tasks.append(task_row)
                                        task_observation.update(
                                            output={
                                                "n_hypotheses": len(hypothesis_rows),
                                                "effective_target": effective_target,
                                                "evidence_pack_summary": task_row["evidence_pack_summary"],
                                            }
                                        )
                            except Exception as exc:
                                failures.append(
                                    {
                                        "run_id": run_id,
                                        "snapshot_id": resolved_snapshot_id,
                                        "method_name": method_name,
                                        "seed": seed,
                                        "target": dict(target),
                                        "target_id": target_label,
                                        "error": str(exc),
                                        "error_type": type(exc).__name__,
                                        "repr": repr(exc),
                                        "traceback": traceback.format_exc(),
                                        "trace_ref": dict(task_trace_ref),
                                    }
                                )

                            task_counter += 1
                            _emit_progress(
                                "generating_tasks",
                                current=task_counter,
                                total=total_tasks,
                                message=f"{method_name} seed={seed} target={target_label}",
                                method_name=method_name,
                                seed=seed,
                                target_label=target_label,
                            )

                run_payload["summary"] = _build_run_summary(tasks, failures)
                run_payload["status"] = "completed_with_failures" if failures else "completed"
                run_observation.update(output=run_payload["summary"])

        _emit_progress("exporting_outputs", message="Writing prospective run outputs")
        output_path = Path(output_dir).expanduser().resolve()
        run_id_text = str(run_payload["run_id"])
        run_payload["output_files"] = {
            "summary_json": str((output_path / f"{run_id_text}_summary.json").resolve()),
            "hypotheses_json": str((output_path / f"{run_id_text}_hypotheses.json").resolve()),
            "hypotheses_csv": str((output_path / f"{run_id_text}_hypotheses.csv").resolve()),
        }
        summary_path, hypotheses_json_path, hypotheses_csv_path = _write_outputs(
            output_path,
            run_payload=run_payload,
            tasks=tasks,
            failures=failures,
        )
        _emit_progress("completed", status="completed", message="Prospective generation completed")
        return ProspectiveResult(
            run=run_payload,
            tasks=tasks,
            failures=failures,
            summary_json=summary_path,
            hypotheses_json=hypotheses_json_path,
            hypotheses_csv=hypotheses_csv_path,
        )
    except Exception as exc:
        run_payload["status"] = "failed"
        run_payload["summary"] = {
            "n_completed_tasks": len(tasks),
            "n_failed_tasks": len(failures),
            "error": str(exc),
        }
        _emit_progress("failed", status="failed", message=f"Prospective generation failed: {exc}")
        raise
    finally:
        flush_langfuse()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prospective hypothesis generation on a published snapshot.")
    parser.add_argument("--backend-url", default="http://localhost:8088")
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--gap-id", action="append", default=None, help="Explicit gap target to run. Repeatable.")
    parser.add_argument(
        "--paper-id",
        action="append",
        default=None,
        help="Paper id to force-include in every evidence pack. Repeatable.",
    )
    parser.add_argument(
        "--required-paper-source-snapshot-id",
        default=None,
        help="Published snapshot from which required paper ids are fetched.",
    )
    parser.add_argument(
        "--cluster-pair",
        action="append",
        nargs=2,
        metavar=("CLUSTER_A", "CLUSTER_B"),
        default=None,
        help="Explicit cluster-pair target to run. Repeatable.",
    )
    parser.add_argument("--n-gap-targets", type=int, default=20)
    parser.add_argument("--n-cluster-pair-targets", type=int, default=10)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=sorted(GENERATOR_REGISTRY))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--hypotheses-per-target", type=int, default=3)
    parser.add_argument("--exemplars", type=int, default=8)
    parser.add_argument("--boundary", type=int, default=8)
    parser.add_argument("--diverse", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--output-dir", default="data/prospective_eval")
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07"))
    parser.add_argument("--discovery-cue-text", default=None)
    parser.add_argument("--discovery-cue-goal", default=None)
    parser.add_argument(
        "--cue-source-snapshot-id",
        default=None,
        help="Snapshot id used for cue-semantic retrieval. Required when discovery cue is active.",
    )
    parser.add_argument("--cue-similarity-top-k", type=int, default=50)
    parser.add_argument("--cue-similarity-sample-n", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reporter = _ProspectiveCliProgressReporter()

    explicit_targets = _explicit_targets(
        gap_ids=args.gap_id,
        cluster_pairs=[(int(pair[0]), int(pair[1])) for pair in (args.cluster_pair or [])],
    )
    if explicit_targets:
        n_gap_targets = 0
        n_cluster_pair_targets = 0
    else:
        n_gap_targets = args.n_gap_targets
        n_cluster_pair_targets = args.n_cluster_pair_targets

    discovery_cue = None
    if args.discovery_cue_text or args.discovery_cue_goal:
        discovery_cue = {
            "text": args.discovery_cue_text or "",
            "goal": args.discovery_cue_goal or None,
        }

    result = run_prospective(
        snapshot_id=args.snapshot_id,
        backend_url=args.backend_url,
        methods=args.methods,
        seeds=args.seeds,
        hypotheses_per_target=args.hypotheses_per_target,
        n_gap_targets=n_gap_targets,
        n_cluster_pair_targets=n_cluster_pair_targets,
        explicit_targets=explicit_targets,
        output_dir=args.output_dir,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name=args.openai_model,
        discovery_cue=discovery_cue,
        cue_source_snapshot_id=args.cue_source_snapshot_id,
        cue_similarity_top_k=args.cue_similarity_top_k,
        cue_similarity_sample_n=args.cue_similarity_sample_n,
        exemplars=args.exemplars,
        boundary=args.boundary,
        diverse=args.diverse,
        max_iters=args.max_iters,
        required_paper_ids=args.paper_id,
        required_paper_source_snapshot_id=args.required_paper_source_snapshot_id,
        progress_callback=reporter,
    )
    print(
        json.dumps(
            {
                "run_id": result.run["run_id"],
                "status": result.run["status"],
                "snapshot_id": result.run["snapshot_id"],
                "summary": result.run["summary"],
                "summary_json": result.summary_json,
                "hypotheses_json": result.hypotheses_json,
                "hypotheses_csv": result.hypotheses_csv,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

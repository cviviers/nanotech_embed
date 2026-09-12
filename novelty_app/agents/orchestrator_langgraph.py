from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any, Dict, List, TypedDict

from pydantic import BaseModel, Field, model_validator

try:
    from agents.backend_client import BackendClient
except Exception:  # pragma: no cover
    try:
        from novelty_app.agents.backend_client import BackendClient  # type: ignore
    except Exception:
        from backend_client import BackendClient  # type: ignore

try:
    from discovery_cue import cue_prompt_block, discovery_cue_to_dict
except Exception:  # pragma: no cover
    from novelty_app.discovery_cue import cue_prompt_block, discovery_cue_to_dict

try:
    from agents.observability import (
        current_trace_ref,
        langchain_config_with_observability,
        observe_current,
        trace_attributes,
    )
except Exception:  # pragma: no cover
    from novelty_app.agents.observability import (
        current_trace_ref,
        langchain_config_with_observability,
        observe_current,
        trace_attributes,
    )

try:
    from evaluation.judge import score_hypotheses
except Exception:  # pragma: no cover
    from novelty_app.evaluation.judge import score_hypotheses

try:
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore
    END = None  # type: ignore
    StateGraph = None  # type: ignore


class Axis(BaseModel):
    axis: str
    what_differs: str
    evidence_A: List[str] = Field(default_factory=list)
    evidence_B: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class BridgeSeed(BaseModel):
    idea: str
    why_plausible: str
    support: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)


class ClusterSummary(BaseModel):
    one_line: str
    bullets: List[str] = Field(default_factory=list)
    salient_entities: Dict[str, List[str]] = Field(default_factory=dict)
    citations: List[str] = Field(default_factory=list)


class ContrastiveExplanation(BaseModel):
    cluster_A_summary: ClusterSummary
    cluster_B_summary: ClusterSummary
    axes_of_separation: List[Axis] = Field(default_factory=list)
    bridge_seeds: List[BridgeSeed] = Field(default_factory=list)
    insufficient_evidence: bool = False


class ClaimSupportStatus(str, Enum):
    supported = "supported"
    partial = "partial"
    unsupported = "unsupported"


def _normalize_claim_support_status(value: Any) -> ClaimSupportStatus:
    if isinstance(value, ClaimSupportStatus):
        return value
    if isinstance(value, bool):
        return ClaimSupportStatus.supported if value else ClaimSupportStatus.unsupported
    text = str(value or "").strip().lower()
    if text in {"supported", "support", "true", "yes", "y"}:
        return ClaimSupportStatus.supported
    if text in {"partial", "partially", "partially supported", "mixed", "somewhat supported"}:
        return ClaimSupportStatus.partial
    if text in {"unsupported", "not supported", "false", "no", "n"}:
        return ClaimSupportStatus.unsupported
    raise ValueError(
        "claim support status must be one of supported, partial, unsupported, or a legacy boolean value"
    )


def _claim_support_weight(value: ClaimSupportStatus) -> float:
    if value == ClaimSupportStatus.supported:
        return 1.0
    if value == ClaimSupportStatus.partial:
        return 0.5
    return 0.0


class AuditClaim(BaseModel):
    claim: str
    support_status: ClaimSupportStatus
    missing_evidence_queries: List[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_support_status(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        raw_status = payload.get("support_status", payload.get("supported"))
        if raw_status is not None:
            payload["support_status"] = _normalize_claim_support_status(raw_status)
        payload.pop("supported", None)
        return payload


class AuditReport(BaseModel):
    supported_claim_fraction: float = Field(ge=0.0, le=1.0)
    needs_patch: bool
    missing_facets: List[str] = Field(default_factory=list)
    patch_queries: List[str] = Field(default_factory=list)
    claims: List[AuditClaim] = Field(default_factory=list)
    cue_alignment_score: float | None = Field(default=None, ge=0.0, le=1.0)
    cue_usage_summary: str = ""
    cue_terms_addressed: List[str] = Field(default_factory=list)
    cue_violations: List[str] = Field(default_factory=list)
    missing_cue_facets: List[str] = Field(default_factory=list)
    respects_hard_constraints: bool = True

    @model_validator(mode="after")
    def _normalize_from_claims(self) -> AuditReport:
        if self.claims:
            self.supported_claim_fraction = sum(
                _claim_support_weight(claim.support_status) for claim in self.claims
            ) / float(len(self.claims))
        has_unsupported_or_partial = any(
            claim.support_status != ClaimSupportStatus.supported for claim in self.claims
        )
        derived_needs_patch = bool(
            has_unsupported_or_partial
            or self.missing_facets
            or self.patch_queries
            or self.cue_violations
            or self.missing_cue_facets
            or not self.respects_hard_constraints
        )
        self.needs_patch = bool(self.needs_patch or derived_needs_patch)
        return self


class Hypothesis(BaseModel):
    id: str
    title: str
    bridge_type: str
    mechanistic_rationale: str
    cue_alignment_rationale: str = ""
    cue_terms_addressed: List[str] = Field(default_factory=list)
    novel_elements: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)


class HypothesesOut(BaseModel):
    hypotheses: List[Hypothesis]


class BlueprintOut(BaseModel):
    bill_of_materials: List[str] = Field(default_factory=list)
    synthesis_and_characterization: Dict[str, Any] = Field(default_factory=dict)
    in_vitro_plan: Dict[str, Any] = Field(default_factory=dict)
    in_vivo_plan: Dict[str, Any] = Field(default_factory=dict)
    risks_and_mitigations: List[Dict[str, Any]] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)


class OrchestratorState(TypedDict, total=False):
    # Keep this as plain str to avoid runtime type-hint evaluation issues across mixed import contexts.
    target_type: str
    snapshot_id: str
    gap_id: str
    cluster_a: int
    cluster_b: int
    max_iters: int
    iter: int
    exemplars: int
    boundary: int
    diverse: int
    required_paper_ids: List[str]
    required_paper_source_snapshot_id: str
    discovery_cue: Dict[str, Any]
    cue_source_snapshot_id: str
    cue_similarity_top_k: int
    cue_similarity_sample_n: int
    cue_similarity_seed: str | int | None
    evidence: List[Dict[str, Any]]
    evidence_meta: Dict[str, Any]
    explanation: Dict[str, Any]
    audit: Dict[str, Any]
    hypotheses: Dict[str, Any]
    idea_scores: Dict[str, Any]
    hypotheses_per_target: int
    source_assessments: Dict[str, Any]
    source_selected_id: str | None
    blueprint: Dict[str, Any]
    published: bool
    published_artifact: Dict[str, Any]
    run_trace_ref: Dict[str, Any]
    observability: Dict[str, Any]


SYSTEM_EXPLAIN = (
    "You are a nanomedicine domain expert. Only use the EVIDENCE PACK provided. "
    "A RESEARCH DIRECTION CUE may be provided as steering context, but it is not evidence. "
    "Never invent facts or cite outside sources. If evidence is insufficient for any claim, say 'unknown'. "
    "Cite by paper_id for every claim/axis. Output strictly valid JSON matching the schema."
)

SYSTEM_AUDIT = (
    "You are a strict scientific auditor. You receive an explanation and an evidence pack. "
    "A RESEARCH DIRECTION CUE may also be provided. The cue is not evidence, but the output should be checked for alignment with it. "
    "Your job is to identify unsupported claims, missing facets, and propose retrieval queries to patch gaps. "
    "When a cue is present, briefly explain how the explanation uses it and list which cue terms or facets are actually addressed. "
    "For each audited claim, use `support_status` with exactly one of: `supported`, `partial`, `unsupported`. "
    "Use `partial` when only part of a claim is directly supported or the support is indirect. "
    "Set `supported_claim_fraction` consistently, counting `partial` as 0.5 support. "
    "Be conservative. Output strictly valid JSON matching the schema."
)

SYSTEM_IDEATE = (
    "You propose testable nanomedicine hypotheses. Use only the evidence pack. "
    "A RESEARCH DIRECTION CUE may be provided to steer direction, but it is not evidence and must not be cited. "
    "Every hypothesis must be testable in 6-12 months by an academic lab. "
    "For each hypothesis, explicitly state how it addresses the cue and list the cue terms or facets it covers. "
    "Cite by paper_id. Output strictly valid JSON matching the schema."
)

SYSTEM_BLUEPRINT = (
    "You produce a concise but complete preclinical blueprint for one hypothesis. "
    "A RESEARCH DIRECTION CUE may be provided to constrain direction, but it is not evidence. "
    "Use only the evidence pack and cite by paper_id. Mark anything not supported as 'assumption'. "
    "Output strictly valid JSON matching the schema."
)


def format_pack_jsonl(papers: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for p in papers:
        lines.append(
            json.dumps(
                {
                    "paper_id": p.get("paper_id"),
                    "title": p.get("title", ""),
                    "year": p.get("year", p.get("publication_year", -1)),
                    "doi": p.get("doi", ""),
                    "cluster_id": p.get("cluster_id", None),
                    "text": (p.get("abstract", "") or p.get("processed_content", ""))[:2000],
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def _target_payload(state: OrchestratorState) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "target_type": state["target_type"],
        "exemplars": state.get("exemplars", 25),
        "boundary": state.get("boundary", 25),
        "diverse": state.get("diverse", 25),
        "counter_queries": [],
    }
    required_paper_ids = [str(paper_id).strip() for paper_id in (state.get("required_paper_ids") or []) if str(paper_id).strip()]
    if required_paper_ids:
        payload["required_paper_ids"] = required_paper_ids
    required_paper_source_snapshot_id = str(state.get("required_paper_source_snapshot_id") or "").strip()
    if required_paper_source_snapshot_id:
        payload["required_paper_source_snapshot_id"] = required_paper_source_snapshot_id
    if state.get("snapshot_id"):
        payload["snapshot_id"] = state["snapshot_id"]
    if state["target_type"] == "gap":
        payload["gap_id"] = state["gap_id"]
    else:
        payload["cluster_a"] = state["cluster_a"]
        payload["cluster_b"] = state["cluster_b"]
    discovery_cue = discovery_cue_to_dict(state.get("discovery_cue"))
    if discovery_cue is not None:
        payload["discovery_cue"] = discovery_cue
        cue_source_snapshot_id = str(state.get("cue_source_snapshot_id") or "").strip()
        if cue_source_snapshot_id:
            payload["cue_source_snapshot_id"] = cue_source_snapshot_id
        if state.get("cue_similarity_top_k") is not None:
            payload["cue_similarity_top_k"] = int(state["cue_similarity_top_k"])
        if state.get("cue_similarity_sample_n") is not None:
            payload["cue_similarity_sample_n"] = int(state["cue_similarity_sample_n"])
        if state.get("cue_similarity_seed") is not None:
            payload["cue_similarity_seed"] = state.get("cue_similarity_seed")
    return payload


def _target_summary(state: OrchestratorState) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "snapshot_id": state.get("snapshot_id"),
        "target_type": state.get("target_type"),
    }
    if state.get("target_type") == "gap":
        summary["gap_id"] = state.get("gap_id")
    else:
        summary["cluster_a"] = state.get("cluster_a")
        summary["cluster_b"] = state.get("cluster_b")
    return {key: value for key, value in summary.items() if value is not None}


def _trace_metadata(state: OrchestratorState) -> Dict[str, Any]:
    return {
        **_target_summary(state),
        "has_discovery_cue": bool(discovery_cue_to_dict(state.get("discovery_cue"))),
        "n_required_paper_ids": len(state.get("required_paper_ids") or []),
        "required_paper_source_snapshot_id": state.get("required_paper_source_snapshot_id"),
        "max_iters": state.get("max_iters"),
    }


def _trace_attribute_metadata(state: OrchestratorState) -> Dict[str, Any]:
    summary = _target_summary(state)
    return {key: str(value) for key, value in summary.items() if value is not None}


def _trace_tags(state: OrchestratorState) -> List[str]:
    tags = ["orchestrator", str(state.get("target_type") or "unknown")]
    if state.get("snapshot_id"):
        tags.append("snapshot")
    if discovery_cue_to_dict(state.get("discovery_cue")):
        tags.append("discovery_cue")
    return tags


def _llm_model_name(llm: ChatOpenAI) -> Optional[str]:
    return getattr(llm, "model_name", None) or getattr(llm, "model", None)


def _langchain_config_for_state(
    state: OrchestratorState,
    *,
    trace_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    session_id = str(state.get("snapshot_id") or "") or None
    return langchain_config_with_observability(
        config,
        session_id=session_id,
        tags=_trace_tags(state),
        trace_name=trace_name,
        metadata=_trace_metadata(state),
    )


def _evidence_summary(papers: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "n_papers": len(papers),
        "paper_ids": [paper.get("paper_id") for paper in papers[:5] if paper.get("paper_id")],
        "profile": (meta or {}).get("profile"),
    }


def _initial_invoke_summary(state: OrchestratorState) -> Dict[str, Any]:
    return {
        **_target_summary(state),
        "max_iters": state.get("max_iters"),
        "exemplars": state.get("exemplars"),
        "boundary": state.get("boundary"),
        "diverse": state.get("diverse"),
        "n_required_paper_ids": len(state.get("required_paper_ids") or []),
        "required_paper_source_snapshot_id": state.get("required_paper_source_snapshot_id"),
        "has_discovery_cue": bool(discovery_cue_to_dict(state.get("discovery_cue"))),
    }


def node_build_pack(state: OrchestratorState, backend: BackendClient) -> OrchestratorState:
    payload = _target_payload(state)
    with observe_current(
        name="build_pack",
        as_type="retriever",
        input_payload=payload,
        metadata=_trace_metadata(state),
    ) as observation:
        pack = backend.evidence_pack(payload)
        state["evidence"] = pack.get("papers", [])
        state["evidence_meta"] = pack.get("meta", {})
        observation.update(output=_evidence_summary(state["evidence"], state["evidence_meta"]))
    return state


def node_explain(state: OrchestratorState, llm: ChatOpenAI) -> OrchestratorState:
    pack = format_pack_jsonl(state.get("evidence", []))
    cue_block = cue_prompt_block(state.get("discovery_cue"))
    if state["target_type"] == "gap":
        ctx = {"gap_id": state["gap_id"], "snapshot_id": state.get("snapshot_id")}
        user = f"""
TASK: Explain what lies on either side of this gap region and propose bridge seeds.
CONTEXT: {json.dumps(ctx)}
{cue_block}
EVIDENCE PACK (JSONL):
```jsonl
{pack}
```
Return JSON per schema.
"""
    else:
        ctx = {
            "cluster_a": state["cluster_a"],
            "cluster_b": state["cluster_b"],
            "snapshot_id": state.get("snapshot_id"),
        }
        user = f"""
TASK: Contrast Cluster A vs Cluster B to explain why they are separated in embedding space.
CONTEXT: {json.dumps(ctx)}
{cue_block}
EVIDENCE PACK (JSONL):
```jsonl
{pack}
```
Return JSON per schema.
"""
    structured = llm.with_structured_output(ContrastiveExplanation, method="function_calling")
    messages = [
        {"role": "system", "content": SYSTEM_EXPLAIN},
        {"role": "user", "content": user},
    ]
    with observe_current(
        name="explain",
        as_type="generation",
        input_payload=messages,
        metadata=_trace_metadata(state),
        model=_llm_model_name(llm),
    ) as observation:
        out = structured.invoke(messages, config=_langchain_config_for_state(state, trace_name="explain"))
        state["explanation"] = out.model_dump()
        observation.update(output=state["explanation"])
    return state


def node_audit(state: OrchestratorState, llm: ChatOpenAI) -> OrchestratorState:
    pack = format_pack_jsonl(state.get("evidence", []))
    expl = json.dumps(state.get("explanation", {}), ensure_ascii=False)
    cue_block = cue_prompt_block(state.get("discovery_cue"))
    user = f"""
INPUT EXPLANATION JSON:
{expl}

{cue_block}
EVIDENCE PACK (JSONL):
```jsonl
{pack}
```
Audit the explanation: identify unsupported claims, missing facets, cue violations, and propose patch retrieval queries.
Also state briefly how the cue is reflected in the explanation and which cue terms or facets are addressed versus still missing.
Return JSON per schema.
"""
    structured = llm.with_structured_output(AuditReport, method="function_calling")
    messages = [
        {"role": "system", "content": SYSTEM_AUDIT},
        {"role": "user", "content": user},
    ]
    with observe_current(
        name="audit",
        as_type="evaluator",
        input_payload=messages,
        metadata=_trace_metadata(state),
        model=_llm_model_name(llm),
    ) as observation:
        out = structured.invoke(messages, config=_langchain_config_for_state(state, trace_name="audit"))
        state["audit"] = out.model_dump(mode="json")
        observation.update(output=state["audit"])
    return state


def node_patch_retrieve(state: OrchestratorState, backend: BackendClient) -> OrchestratorState:
    audit = state.get("audit", {})
    queries = (audit.get("patch_queries") or [])[:8]
    payload = _target_payload(state)
    payload["exemplars"] = max(10, state.get("exemplars", 25) // 2)
    payload["boundary"] = max(10, state.get("boundary", 25) // 2)
    payload["diverse"] = max(10, state.get("diverse", 25) // 2)
    payload["counter_queries"] = queries

    with observe_current(
        name="patch_retrieve",
        as_type="retriever",
        input_payload={"payload": payload, "patch_queries": queries},
        metadata=_trace_metadata(state),
    ) as observation:
        papers = backend.evidence_pack(payload).get("papers", [])
        seen = {p.get("paper_id") for p in state.get("evidence", [])}
        merged = list(state.get("evidence", []))
        added = 0
        for p in papers:
            if p.get("paper_id") not in seen:
                merged.append(p)
                seen.add(p.get("paper_id"))
                added += 1
        state["evidence"] = merged
        state["iter"] = state.get("iter", 0) + 1
        observation.update(
            output={
                "patch_queries": queries,
                "added_papers": added,
                "evidence_size": len(state["evidence"]),
            }
        )
    return state


def node_ideate(state: OrchestratorState, llm: ChatOpenAI) -> OrchestratorState:
    pack = format_pack_jsonl(state.get("evidence", []))
    expl = json.dumps(state.get("explanation", {}), ensure_ascii=False)
    cue_block = cue_prompt_block(state.get("discovery_cue"))
    user = f"""
GOAL: Propose {state.get('hypotheses_per_target', 5)} bridge hypotheses grounded in the evidence pack and the contrastive explanation.
EXPLANATION JSON:
{expl}

{cue_block}
EVIDENCE PACK (JSONL):
```jsonl
{pack}
```
For each hypothesis, include `cue_alignment_rationale` and `cue_terms_addressed`.
Return JSON per schema.
"""
    structured = llm.with_structured_output(HypothesesOut, method="function_calling")
    messages = [
        {"role": "system", "content": SYSTEM_IDEATE},
        {"role": "user", "content": user},
    ]
    with observe_current(
        name="ideate",
        as_type="generation",
        input_payload=messages,
        metadata=_trace_metadata(state),
        model=_llm_model_name(llm),
    ) as observation:
        out = structured.invoke(messages, config=_langchain_config_for_state(state, trace_name="ideate"))
        state["hypotheses"] = out.model_dump()
        observation.update(output=state["hypotheses"])
    return state


def node_score(
    state: OrchestratorState,
    *,
    openai_api_key: str | None = None,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
) -> OrchestratorState:
    hypotheses = list((state.get("hypotheses") or {}).get("hypotheses") or [])
    with observe_current(
        name="score",
        as_type="evaluator",
        input_payload={
            "hypotheses": [{"id": hyp.get("id"), "title": hyp.get("title")} for hyp in hypotheses],
            "target": _target_summary(state),
            "evidence_size": len(state.get("evidence", [])),
        },
        metadata=_trace_metadata(state),
    ) as observation:
        scores = score_hypotheses(
            hypotheses,
            evidence_pack={
                "papers": state.get("evidence", []),
                "meta": state.get("evidence_meta", {}),
            },
            audit=state.get("audit", {}),
            explanation=state.get("explanation", {}),
            target={
                "target_type": state.get("target_type"),
                "gap_id": state.get("gap_id"),
                "cluster_a": state.get("cluster_a"),
                "cluster_b": state.get("cluster_b"),
                "snapshot_id": state.get("snapshot_id"),
            },
            discovery_cue=discovery_cue_to_dict(state.get("discovery_cue")),
            openai_api_key=openai_api_key,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
        )
        state["idea_scores"] = scores
        for hyp in hypotheses:
            hyp_id = str(hyp.get("id") or "")
            hyp["idea_scores"] = dict(scores.get(hyp_id) or {})
        state["hypotheses"] = {"hypotheses": hypotheses}
        observation.update(output={"scores": scores})
    return state


def node_blueprint(state: OrchestratorState, llm: ChatOpenAI) -> OrchestratorState:
    pack = format_pack_jsonl(state.get("evidence", []))
    hyps = state.get("hypotheses", {}).get("hypotheses", [])
    if not hyps:
        state["blueprint"] = {}
        return state
    scores = state.get("idea_scores", {})
    ranked_hyps = sorted(
        hyps,
        key=lambda hyp: float(
            (scores.get(str(hyp.get("id") or ""), {}) or {}).get("average_score")
            or (hyp.get("idea_scores") or {}).get("average_score")
            or 0.0
        ),
        reverse=True,
    )
    if "source_selected_id" in state:
        chosen = state["source_selected_id"]
        if chosen is None:
            state["blueprint"] = {}
            return state
        h1 = next(h for h in hyps if str(h.get("id")) == chosen)
    else:
        h1 = ranked_hyps[0]
    cue_block = cue_prompt_block(state.get("discovery_cue"))
    user = f"""
HYPOTHESIS JSON:
{json.dumps(h1, ensure_ascii=False)}

{cue_block}
EVIDENCE PACK (JSONL):
```jsonl
{pack}
```
Return JSON per schema.
"""
    structured = llm.with_structured_output(BlueprintOut, method="function_calling")
    messages = [
        {"role": "system", "content": SYSTEM_BLUEPRINT},
        {"role": "user", "content": user},
    ]
    with observe_current(
        name="blueprint",
        as_type="generation",
        input_payload=messages,
        metadata=_trace_metadata(state),
        model=_llm_model_name(llm),
    ) as observation:
        out = structured.invoke(messages, config=_langchain_config_for_state(state, trace_name="blueprint"))
        state["blueprint"] = out.model_dump()
        observation.update(output=state["blueprint"])
    return state


def node_publish(state: OrchestratorState, backend: BackendClient) -> OrchestratorState:
    target: Dict[str, Any] = {"target_type": state["target_type"]}
    if state.get("snapshot_id"):
        target["snapshot_id"] = state["snapshot_id"]
    if state["target_type"] == "gap":
        target["gap_id"] = state["gap_id"]
    else:
        target["cluster_a"] = state["cluster_a"]
        target["cluster_b"] = state["cluster_b"]

    payload = {
        "evidence_size": len(state.get("evidence", [])),
        "evidence": list(state.get("evidence", [])),
        "discovery_cue": discovery_cue_to_dict(state.get("discovery_cue")),
        "evidence_meta": state.get("evidence_meta", {}),
        "explanation": state.get("explanation", {}),
        "audit": state.get("audit", {}),
        "hypotheses": state.get("hypotheses", {}),
        "idea_scores": state.get("idea_scores", {}),
        "source_assessments": state.get("source_assessments", {}),
        "source_selected_id": state.get("source_selected_id"),
        "blueprint": state.get("blueprint", {}),
        "iterations": state.get("iter", 0),
        "trace_ref": dict(state.get("run_trace_ref") or {}),
    }
    with observe_current(
        name="publish",
        as_type="tool",
        input_payload={"target": target, "payload": {"evidence_size": payload["evidence_size"], "iterations": payload["iterations"]}},
        metadata=_trace_metadata(state),
    ) as observation:
        artifact = backend.store_artifact(kind="research_brief", target=target, payload=payload)
        state["published_artifact"] = artifact
        state["published"] = True
        observation.update(output={"artifact": artifact, "trace_ref": payload["trace_ref"]})
    return state


def route_after_audit(state: OrchestratorState) -> str:
    audit = state.get("audit", {})
    needs_patch = bool(audit.get("needs_patch", False))
    cue_alignment = audit.get("cue_alignment_score")
    it = state.get("iter", 0)
    max_iters = state.get("max_iters", 2)
    if (needs_patch or (cue_alignment is not None and float(cue_alignment) < 0.35)) and it < max_iters:
        return "patch"
    return "ideate"


class InstrumentedOrchestrator:
    def __init__(self, compiled: Any):
        self._compiled = compiled

    def __getattr__(self, name: str) -> Any:
        return getattr(self._compiled, name)

    def invoke(self, state: OrchestratorState, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        seeded_state = dict(state)
        session_id = str(seeded_state.get("snapshot_id") or "") or None
        tags = _trace_tags(seeded_state)
        metadata = _trace_metadata(seeded_state)
        with observe_current(
            name="orchestrator_run",
            as_type="agent",
            input_payload=_initial_invoke_summary(seeded_state),
            metadata=metadata,
        ) as observation:
            with trace_attributes(
                session_id=session_id,
                tags=tags,
                trace_name="orchestrator_run",
                metadata=_trace_attribute_metadata(seeded_state),
            ):
                seeded_state["run_trace_ref"] = current_trace_ref(
                    session_id=session_id,
                    tags=tags,
                    metadata=metadata,
                )
                graph_kwargs = dict(kwargs)
                graph_kwargs["config"] = _langchain_config_for_state(
                    seeded_state,
                    trace_name="orchestrator_run",
                    config=graph_kwargs.get("config"),
                )
                out = dict(self._compiled.invoke(seeded_state, *args, **graph_kwargs))
                out["observability"] = current_trace_ref(
                    session_id=session_id,
                    tags=tags,
                    metadata=metadata,
                )
                observation.update(
                    output={
                        "published": bool(out.get("published")),
                        "iterations": out.get("iter", 0),
                        "evidence_size": len(out.get("evidence", [])),
                        "artifact_id": (out.get("published_artifact") or {}).get("artifact_id"),
                    }
                )
                return out


def build_orchestrator(
    backend: BackendClient,
    *,
    openai_api_key: str | None = None,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
    ideation_model: str | None = None,
    assessment_callback: Any = None,
) -> Any:
    if ChatOpenAI is None or StateGraph is None or END is None:
        raise ImportError(
            "langchain-openai and langgraph are required to build the orchestrator. "
            "Install them in the active environment first."
        )

    model = model_name or os.getenv("OPENAI_MODEL", "gpt-5")
    llm_kwargs: Dict[str, Any] = {"model": model, "temperature": 0.2}
    if reasoning_effort:
        llm_kwargs = {"model": model, "reasoning_effort": reasoning_effort}
    ideate_llm_kwargs: Dict[str, Any] = dict(llm_kwargs, model=ideation_model or model)
    if openai_api_key:
        llm_kwargs["api_key"] = openai_api_key
        ideate_llm_kwargs["api_key"] = openai_api_key
    llm = ChatOpenAI(**llm_kwargs)
    ideate_llm = ChatOpenAI(**ideate_llm_kwargs)
    # type: ignore[arg-type]
    g = StateGraph(OrchestratorState)

    g.add_node("build_pack", lambda s: node_build_pack(s, backend))
    g.add_node("explain", lambda s: node_explain(s, llm))
    g.add_node("audit", lambda s: node_audit(s, llm))
    g.add_node("patch_retrieve", lambda s: node_patch_retrieve(s, backend))
    g.add_node("ideate", lambda s: node_ideate(s, ideate_llm))
    g.add_node("score", lambda s: node_score(s, openai_api_key=openai_api_key, model_name=model, reasoning_effort=reasoning_effort))
    g.add_node("blueprint", lambda s: node_blueprint(s, llm))
    g.add_node("publish", lambda s: node_publish(s, backend))

    g.set_entry_point("build_pack")
    g.add_edge("build_pack", "explain")
    g.add_edge("explain", "audit")
    g.add_conditional_edges("audit", route_after_audit, {"patch": "patch_retrieve", "ideate": "ideate"})
    g.add_edge("patch_retrieve", "explain")
    g.add_edge("ideate", "score")
    if assessment_callback is not None:
        g.add_node("source_assessment", assessment_callback)
        g.add_edge("score", "source_assessment")
        g.add_edge("source_assessment", "blueprint")
    else:
        g.add_edge("score", "blueprint")
    g.add_edge("blueprint", "publish")
    g.add_edge("publish", END)

    return InstrumentedOrchestrator(g.compile())


__all__ = [
    "BackendClient",
    "OrchestratorState",
    "build_orchestrator",
    "route_after_audit",
]

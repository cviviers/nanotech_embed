from __future__ import annotations

import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

try:
    from agents.backend_client import BackendClient
    from agents.orchestrator_langgraph import (
        HypothesesOut,
        SYSTEM_IDEATE,
        build_orchestrator,
        format_pack_jsonl,
    )
    from agents.schemas import GeneratedHypothesis
except Exception:  # pragma: no cover
    from novelty_app.agents.backend_client import BackendClient
    from novelty_app.agents.orchestrator_langgraph import (
        HypothesesOut,
        SYSTEM_IDEATE,
        build_orchestrator,
        format_pack_jsonl,
    )
    from novelty_app.agents.schemas import GeneratedHypothesis

from novelty_app.discovery_cue import cue_prompt_block, discovery_cue_to_dict, normalize_discovery_cue
from novelty_app.agents.observability import (
    current_trace_ref,
    langchain_config_with_observability,
    observe_current,
)

from .idea_fingerprint import fingerprint_hypothesis, fingerprint_text


@dataclass
class GenerationContext:
    backend: BackendClient
    snapshot_id: str
    target: Dict[str, Any]
    seed: int = 0
    openai_api_key: Optional[str] = None
    model_name: Optional[str] = None
    discovery_cue: Optional[Dict[str, Any]] = None
    cue_source_snapshot_id: Optional[str] = None
    cue_similarity_top_k: int = 50
    cue_similarity_sample_n: int = 6
    cue_similarity_seed: Optional[str | int] = None
    exemplars: int = 8
    boundary: int = 8
    diverse: int = 0
    required_paper_ids: Optional[Sequence[str]] = None
    required_paper_source_snapshot_id: Optional[str] = None
    evidence_pack_profile: str = "focused_eval"
    max_iters: int = 2
    hypotheses_per_target: int = 3
    all_clusters: Optional[Sequence[int]] = None
    all_targets: Optional[Sequence[Dict[str, Any]]] = None
    reasoning_effort: Optional[str] = None
    frozen_evidence_pack: Optional[Dict[str, Any]] = None


def target_id(target: Dict[str, Any]) -> str:
    if target.get("target_type") == "gap":
        return str(target.get("gap_id"))
    return f"cluster_pair_{target.get('cluster_a')}_{target.get('cluster_b')}"


def _pack_request(context: GenerationContext, target_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    target = dict(target_override or context.target)
    required_paper_ids = [
        paper_id
        for paper_id in dict.fromkeys(str(value or "").strip() for value in (context.required_paper_ids or []))
        if paper_id
    ]
    payload: Dict[str, Any] = {
        "snapshot_id": context.snapshot_id,
        "target_type": target["target_type"],
        "profile": context.evidence_pack_profile,
        "exemplars": context.exemplars,
        "boundary": context.boundary,
        "diverse": context.diverse,
        "counter_queries": [],
    }
    if required_paper_ids:
        payload["required_paper_ids"] = required_paper_ids
    if context.required_paper_source_snapshot_id:
        payload["required_paper_source_snapshot_id"] = str(context.required_paper_source_snapshot_id)
    discovery_cue = discovery_cue_to_dict(context.discovery_cue)
    if discovery_cue is not None:
        payload["discovery_cue"] = discovery_cue
        if context.cue_source_snapshot_id:
            payload["cue_source_snapshot_id"] = str(context.cue_source_snapshot_id)
        payload["cue_similarity_top_k"] = int(context.cue_similarity_top_k)
        payload["cue_similarity_sample_n"] = int(context.cue_similarity_sample_n)
        if context.cue_similarity_seed is not None:
            payload["cue_similarity_seed"] = context.cue_similarity_seed
    if target["target_type"] == "gap":
        payload["gap_id"] = target["gap_id"]
    else:
        payload["cluster_a"] = int(target["cluster_a"])
        payload["cluster_b"] = int(target["cluster_b"])
    return payload


def _cluster_only_pack_request(context: GenerationContext) -> Dict[str, Any]:
    payload = _pack_request(context)
    payload["boundary"] = 0
    return payload


def _cue_retrieval_pack_request(context: GenerationContext) -> Dict[str, Any]:
    discovery_cue = discovery_cue_to_dict(context.discovery_cue)
    if discovery_cue is None:
        raise ValueError("cue_retrieval_generation requires an active discovery cue")
    if not context.cue_source_snapshot_id:
        raise ValueError("cue_retrieval_generation requires cue_source_snapshot_id")

    payload: Dict[str, Any] = {
        "snapshot_id": context.snapshot_id,
        "target_type": "cue",
        "profile": "focused_eval",
        "exemplars": 0,
        "boundary": 0,
        "diverse": 0,
        "counter_queries": [],
        "discovery_cue": discovery_cue,
        "cue_source_snapshot_id": str(context.cue_source_snapshot_id),
        "cue_similarity_top_k": int(context.cue_similarity_top_k),
        "cue_similarity_sample_n": int(context.cue_similarity_sample_n),
    }
    return payload


def _random_control_target(context: GenerationContext) -> Dict[str, Any]:
    candidates = [dict(target) for target in (context.all_targets or [])]
    current_target_id = target_id(context.target)
    candidates = [target for target in candidates if target_id(target) != current_target_id]
    if candidates:
        rng = random.Random(context.seed)
        return dict(rng.choice(candidates))

    clusters = [int(c) for c in (context.all_clusters or [])]
    if len(clusters) < 2:
        clusters = [
            int(row["cluster_id"])
            for row in context.backend.list_clusters(snapshot_id=context.snapshot_id, limit=50).get("clusters", [])
        ]
    if len(clusters) < 2:
        raise ValueError("random cluster-pair control requires at least two clusters")
    rng = random.Random(context.seed)
    cluster_a, cluster_b = rng.sample(sorted(set(clusters)), 2)
    return {"target_type": "cluster_pair", "cluster_a": cluster_a, "cluster_b": cluster_b}


def _llm() -> ChatOpenAI:
    if ChatOpenAI is None:
        raise ImportError("langchain-openai is required for LLM generation methods")
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.2,
    )


def _llm_from_context(context: GenerationContext) -> ChatOpenAI:
    if ChatOpenAI is None:
        raise ImportError("langchain-openai is required for LLM generation methods")
    kwargs: Dict[str, Any] = {
        "model": context.model_name or os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07"),
        "temperature": 0.2,
    }
    api_key = context.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for this generation method")
    kwargs["api_key"] = api_key
    if context.reasoning_effort:
        kwargs.pop("temperature", None)
        kwargs["reasoning_effort"] = context.reasoning_effort
    return ChatOpenAI(**kwargs)


def _as_generated_hypotheses(
    raw_hypotheses: Sequence[Dict[str, Any]],
    *,
    context: GenerationContext,
    method_name: str,
    target: Optional[Dict[str, Any]] = None,
    grounding_summary: Optional[Dict[str, Any]] = None,
) -> List[GeneratedHypothesis]:
    out: List[GeneratedHypothesis] = []
    effective_target = target or context.target
    this_target_id = target_id(effective_target)
    trace_ref = current_trace_ref(
        session_id=context.snapshot_id,
        tags=[method_name, str(effective_target.get("target_type") or "unknown")],
        metadata={"snapshot_id": context.snapshot_id, "target_id": this_target_id},
    )
    for idx, raw in enumerate(raw_hypotheses[: context.hypotheses_per_target]):
        title = str(raw.get("title") or raw.get("idea") or f"Hypothesis {idx + 1}")
        text = str(raw.get("mechanistic_rationale") or raw.get("why_plausible") or title)
        model = GeneratedHypothesis(
            hypothesis_id=str(raw.get("id") or f"{method_name}_{context.seed}_{idx}"),
            target_id=this_target_id,
            target_type=str(effective_target.get("target_type") or "unknown"),
            method_name=method_name,
            model_name=context.model_name or os.getenv("OPENAI_MODEL"),
            seed=context.seed,
            title=title,
            text=text,
            support_citations=list(raw.get("citations") or raw.get("support") or []),
            grounding_summary=dict(grounding_summary or {}),
            raw_hypothesis=dict(raw),
            normalized_hypothesis={"title": title, "text": text},
            discovery_cue=dict(discovery_cue_to_dict(context.discovery_cue) or {}),
            trace_ref=trace_ref or {},
        )
        model.idea_fingerprint = fingerprint_hypothesis(model.model_dump())
        out.append(model)
    return out


def _trace_metadata(context: GenerationContext, *, method_name: str, target: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    effective_target = dict(target or context.target)
    metadata: Dict[str, Any] = {
        "snapshot_id": context.snapshot_id,
        "method_name": method_name,
        "target_type": effective_target.get("target_type"),
        "seed": context.seed,
        "hypotheses_per_target": context.hypotheses_per_target,
    }
    if effective_target.get("target_type") == "gap":
        metadata["gap_id"] = effective_target.get("gap_id")
    else:
        metadata["cluster_a"] = effective_target.get("cluster_a")
        metadata["cluster_b"] = effective_target.get("cluster_b")
    return metadata


def _pack_summary(pack: Dict[str, Any]) -> Dict[str, Any]:
    papers = list(pack.get("papers") or [])
    return {
        "snapshot_id": pack.get("snapshot_id"),
        "target_type": pack.get("target_type"),
        "n_papers": len(papers),
        "paper_ids": [paper.get("paper_id") for paper in papers[:5] if paper.get("paper_id")],
        "profile": (pack.get("meta") or {}).get("profile"),
    }


def _llm_name(llm: ChatOpenAI) -> Optional[str]:
    return getattr(llm, "model_name", None) or getattr(llm, "model", None)


def _langchain_config(
    context: GenerationContext,
    *,
    method_name: str,
    target: Optional[Dict[str, Any]] = None,
    trace_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    effective_target = dict(target or context.target)
    target_tags = [method_name, str(effective_target.get("target_type") or "unknown")]
    return langchain_config_with_observability(
        config,
        session_id=context.snapshot_id,
        tags=target_tags,
        trace_name=trace_name or method_name,
        metadata=_trace_metadata(context, method_name=method_name, target=effective_target),
    )


def generate_with_orchestrator(context: GenerationContext) -> Tuple[List[GeneratedHypothesis], Dict[str, Any]]:
    if context.frozen_evidence_pack is not None:
        raise ValueError("Adaptive orchestration cannot replay frozen packs; use the controlled contrastive condition")
    app = build_orchestrator(
        context.backend,
        openai_api_key=context.openai_api_key or os.getenv("OPENAI_API_KEY"),
        model_name=context.model_name,
        reasoning_effort=context.reasoning_effort,
    )
    state: Dict[str, Any] = {
        "snapshot_id": context.snapshot_id,
        "target_type": context.target["target_type"],
        "max_iters": context.max_iters,
        "iter": 0,
        "exemplars": context.exemplars,
        "boundary": context.boundary,
        "diverse": context.diverse,
        "discovery_cue": discovery_cue_to_dict(context.discovery_cue),
        "cue_source_snapshot_id": context.cue_source_snapshot_id,
        "cue_similarity_top_k": context.cue_similarity_top_k,
        "cue_similarity_sample_n": context.cue_similarity_sample_n,
        "cue_similarity_seed": context.cue_similarity_seed,
        "required_paper_ids": list(context.required_paper_ids or []),
        "required_paper_source_snapshot_id": context.required_paper_source_snapshot_id,
        "hypotheses_per_target": context.hypotheses_per_target,
    }
    if context.target["target_type"] == "gap":
        state["gap_id"] = context.target["gap_id"]
    else:
        state["cluster_a"] = int(context.target["cluster_a"])
        state["cluster_b"] = int(context.target["cluster_b"])
    out = app.invoke(state)
    hypotheses = out.get("hypotheses", {}).get("hypotheses", [])
    generated = _as_generated_hypotheses(
        hypotheses,
        context=context,
        method_name="orchestrator",
        grounding_summary=out.get("audit", {}),
    )
    meta = {
        "audit": out.get("audit", {}),
        "explanation": out.get("explanation", {}),
        "blueprint": out.get("blueprint", {}),
        "evidence_pack": {
            "snapshot_id": context.snapshot_id,
            "target_type": context.target["target_type"],
            "papers": out.get("evidence", []),
            "meta": out.get("evidence_meta", {}),
        },
        "effective_target": dict(context.target),
        "published": bool(out.get("published")),
        "published_artifact": dict(out.get("published_artifact") or {}),
        "observability": dict(out.get("observability") or {}),
        "iterations": int(out.get("iter", 0) or 0),
    }
    return generated, meta


def _single_shot_from_pack(
    context: GenerationContext,
    *,
    method_name: str,
    pack_payload: Dict[str, Any],
    preamble: str = "",
    goal: Optional[str] = None,
    target_override: Optional[Dict[str, Any]] = None,
    supplied_pack: Optional[Dict[str, Any]] = None,
) -> Tuple[List[GeneratedHypothesis], Dict[str, Any]]:
    with observe_current(
        name=f"{method_name}_evidence_pack",
        as_type="retriever",
        input_payload=pack_payload,
        metadata=_trace_metadata(context, method_name=method_name, target=target_override),
    ) as pack_observation:
        backend_pack = supplied_pack if supplied_pack is not None else context.frozen_evidence_pack
        if backend_pack is None:
            backend_pack = context.backend.evidence_pack(pack_payload)
        pack_observation.update(output=_pack_summary(backend_pack))
    papers = backend_pack.get("papers", [])
    llm = _llm_from_context(context)
    structured = llm.with_structured_output(HypothesesOut, method="function_calling")
    cue_block = cue_prompt_block(context.discovery_cue)
    generation_goal = goal or f"Propose {context.hypotheses_per_target} grounded nanomedicine bridge hypotheses."
    prompt = f"""
GOAL: {generation_goal}
Only use the EVIDENCE PACK provided. Cite by paper_id.
{cue_block}
{preamble}

EVIDENCE PACK (JSONL):
```jsonl
{format_pack_jsonl(papers)}
```
"""
    messages = [
        {"role": "system", "content": SYSTEM_IDEATE},
        {"role": "user", "content": prompt},
    ]
    with observe_current(
        name=method_name,
        as_type="generation",
        input_payload=messages,
        metadata=_trace_metadata(context, method_name=method_name, target=target_override),
        model=_llm_name(llm),
    ) as generation_observation:
        out = structured.invoke(
            messages,
            config=_langchain_config(
                context,
                method_name=method_name,
                target=target_override,
                trace_name=method_name,
            ),
        )
        generation_observation.update(output=out.model_dump())
    generated = _as_generated_hypotheses(
        out.model_dump().get("hypotheses", []),
        context=context,
        method_name=method_name,
        target=target_override,
        grounding_summary={"n_evidence_papers": len(papers)},
    )
    return generated, {"evidence_pack": backend_pack, "effective_target": dict(target_override or context.target)}


def generate_single_shot_llm(context: GenerationContext) -> Tuple[List[GeneratedHypothesis], Dict[str, Any]]:
    return _single_shot_from_pack(
        context,
        method_name="single_shot_llm",
        pack_payload=_pack_request(context),
    )


def generate_cue_retrieval_generation(
    context: GenerationContext,
) -> Tuple[List[GeneratedHypothesis], Dict[str, Any]]:
    return _single_shot_from_pack(
        context,
        method_name="cue_retrieval_generation",
        pack_payload=_cue_retrieval_pack_request(context),
        goal=(
            f"Use the research-direction cue as the retrieval query and propose "
            f"{context.hypotheses_per_target} grounded, testable nanomedicine hypotheses."
        ),
        preamble=(
            "This is a retrieval-first baseline. The evidence was selected from the historical corpus "
            "using only the research-direction cue. Do not infer or mention any graph frontier, gap, "
            "cluster, or held-out paper."
        ),
    )


def generate_retrieval_summary_direct(context: GenerationContext) -> Tuple[List[GeneratedHypothesis], Dict[str, Any]]:
    pack_payload = _pack_request(context)
    with observe_current(
        name="retrieval_summary_direct_evidence_pack",
        as_type="retriever",
        input_payload=pack_payload,
        metadata=_trace_metadata(context, method_name="retrieval_summary_direct"),
    ) as pack_observation:
        backend_pack = context.frozen_evidence_pack
        if backend_pack is None:
            backend_pack = context.backend.evidence_pack(pack_payload)
        pack_observation.update(output=_pack_summary(backend_pack))
    papers = backend_pack.get("papers", [])
    llm = _llm_from_context(context)
    cue_block = cue_prompt_block(context.discovery_cue)
    summary_prompt = f"""
Summarize the main bridgeable differences and opportunities in the evidence pack.
Return 6 concise bullets grounded only in the cited papers.

{cue_block}
EVIDENCE PACK (JSONL):
```jsonl
{format_pack_jsonl(papers)}
```
"""
    messages = [
        {"role": "system", "content": "You are a careful scientific summarizer. Use only the evidence pack."},
        {"role": "user", "content": summary_prompt},
    ]
    with observe_current(
        name="retrieval_summary_direct_summary",
        as_type="generation",
        input_payload=messages,
        metadata=_trace_metadata(context, method_name="retrieval_summary_direct"),
        model=_llm_name(llm),
    ) as generation_observation:
        summary = llm.invoke(
            messages,
            config=_langchain_config(
                context,
                method_name="retrieval_summary_direct",
                trace_name="retrieval_summary_direct_summary",
            ),
        ).content
        generation_observation.update(output={"summary": summary})
    return _single_shot_from_pack(
        context,
        method_name="retrieval_summary_direct",
        pack_payload=_pack_request(context),
        preamble=f"Use this retrieval-only summary as context:\n{summary}",
        supplied_pack=backend_pack,
    )


def generate_cluster_only(context: GenerationContext) -> Tuple[List[GeneratedHypothesis], Dict[str, Any]]:
    try:
        return _single_shot_from_pack(
            context,
            method_name="cluster_only",
            pack_payload=_cluster_only_pack_request(context),
            preamble="Do not assume any gap-region evidence beyond the cluster exemplars and diverse support.",
        )
    except Exception:
        return generate_heuristic_bridge(context, method_name="cluster_only")


def _heuristic_terms_from_pack(papers: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    counters = {
        "material": Counter(),
        "disease": Counter(),
        "targeting": Counter(),
        "payload": Counter(),
        "mechanism": Counter(),
    }
    for paper in papers:
        fp = fingerprint_text(f"{paper.get('title', '')} {paper.get('abstract', '')}")
        for field in counters:
            for value in fp.get(field, []):
                counters[field][value] += 1
    return {field: [term for term, _count in counter.most_common(5)] for field, counter in counters.items()}


def generate_heuristic_bridge(
    context: GenerationContext,
    *,
    method_name: str = "heuristic_bridge",
    target_override: Optional[Dict[str, Any]] = None,
) -> Tuple[List[GeneratedHypothesis], Dict[str, Any]]:
    effective_target = target_override or context.target
    pack_payload = _pack_request(context, target_override=effective_target)
    with observe_current(
        name=f"{method_name}_evidence_pack",
        as_type="retriever",
        input_payload=pack_payload,
        metadata=_trace_metadata(context, method_name=method_name, target=effective_target),
    ) as pack_observation:
        pack = context.backend.evidence_pack(pack_payload)
        pack_observation.update(output=_pack_summary(pack))
    papers = pack.get("papers", [])
    terms = _heuristic_terms_from_pack(papers)
    cue = normalize_discovery_cue(context.discovery_cue)
    if cue is not None:
        for field in ("material", "disease", "targeting", "payload", "mechanism"):
            cue_terms = list((cue.fingerprint or {}).get(field) or [])
            if cue_terms:
                terms[field] = list(dict.fromkeys(cue_terms + terms.get(field, [])))
    support = [str(p.get("paper_id")) for p in papers[:5]]
    materials = terms.get("material") or ["nanoparticle"]
    diseases = terms.get("disease") or ["cancer"]
    ligands = terms.get("targeting") or ["targeting ligand"]
    payloads = terms.get("payload") or ["drug"]
    mechanisms = terms.get("mechanism") or ["delivery"]
    generated: List[GeneratedHypothesis] = []
    for idx in range(context.hypotheses_per_target):
        title = f"{materials[idx % len(materials)].title()} bridge for {diseases[idx % len(diseases)]}"
        text = (
            f"Combine {materials[idx % len(materials)]} with {ligands[idx % len(ligands)]} targeting "
            f"to improve {payloads[idx % len(payloads)]} {mechanisms[idx % len(mechanisms)]} in {diseases[idx % len(diseases)]}."
        )
        hypothesis = GeneratedHypothesis(
            hypothesis_id=f"{method_name}_{context.seed}_{idx}",
            target_id=target_id(effective_target),
            target_type=str(effective_target.get("target_type") or "unknown"),
            method_name=method_name,
            model_name="heuristic",
            seed=context.seed,
            title=title,
            text=text,
            support_citations=support,
            grounding_summary={"n_evidence_papers": len(papers), "heuristic": True},
            raw_hypothesis={"title": title, "mechanistic_rationale": text, "citations": support},
            normalized_hypothesis={"title": title, "text": text},
            discovery_cue=dict(discovery_cue_to_dict(context.discovery_cue) or {}),
        )
        hypothesis.idea_fingerprint = fingerprint_hypothesis(hypothesis.model_dump())
        generated.append(hypothesis)
    return generated, {"evidence_pack": pack, "effective_target": dict(effective_target)}


def generate_pack_query_baseline(context: GenerationContext) -> Tuple[List[GeneratedHypothesis], Dict[str, Any]]:
    pack_payload = _pack_request(context)
    with observe_current(
        name="pack_query_baseline_evidence_pack",
        as_type="retriever",
        input_payload=pack_payload,
        metadata=_trace_metadata(context, method_name="pack_query_baseline"),
    ) as pack_observation:
        pack = context.backend.evidence_pack(pack_payload)
        pack_observation.update(output=_pack_summary(pack))
    papers = pack.get("papers", [])
    terms = _heuristic_terms_from_pack(papers)
    cue = normalize_discovery_cue(context.discovery_cue)
    if cue is not None:
        for field in ("material", "disease", "targeting", "payload", "mechanism"):
            cue_terms = list((cue.fingerprint or {}).get(field) or [])
            if cue_terms:
                terms[field] = list(dict.fromkeys(cue_terms + terms.get(field, [])))

    material = (terms.get("material") or ["nanoparticle"])[0]
    disease = (terms.get("disease") or ["cancer"])[0]
    payload = (terms.get("payload") or ["drug"])[0]
    targeting = (terms.get("targeting") or ["targeting"])[0]
    mechanism = (terms.get("mechanism") or ["delivery"])[0]
    query_text = " ".join([material, targeting, payload, disease, mechanism]).strip()
    support = [str(p.get("paper_id")) for p in papers[:5]]

    generated: List[GeneratedHypothesis] = []
    for idx in range(context.hypotheses_per_target):
        title = f"Pack-query baseline {idx + 1}: {query_text}"
        text = (
            f"Investigate a {material}-based {targeting} system for {payload} {mechanism} in {disease}. "
            f"Use this retrieval-oriented query: {query_text}."
        )
        hypothesis = GeneratedHypothesis(
            hypothesis_id=f"pack_query_baseline_{context.seed}_{idx}",
            target_id=target_id(context.target),
            target_type=str(context.target.get("target_type") or "unknown"),
            method_name="pack_query_baseline",
            model_name="deterministic",
            seed=context.seed,
            title=title,
            text=text,
            support_citations=support,
            grounding_summary={"n_evidence_papers": len(papers), "deterministic_pack_query": True},
            raw_hypothesis={"title": title, "mechanistic_rationale": text, "citations": support},
            normalized_hypothesis={"title": title, "text": text},
            discovery_cue=dict(discovery_cue_to_dict(context.discovery_cue) or {}),
        )
        hypothesis.idea_fingerprint = fingerprint_hypothesis(
            {
                **hypothesis.model_dump(),
                "text": query_text,
            }
        )
        hypothesis.idea_fingerprint["query_text"] = query_text
        generated.append(hypothesis)
    return generated, {"evidence_pack": pack, "effective_target": dict(context.target)}


def generate_random_target_control(context: GenerationContext) -> Tuple[List[GeneratedHypothesis], Dict[str, Any]]:
    random_target = _random_control_target(context)
    try:
        return _single_shot_from_pack(
            context,
            method_name="random_target_control",
            pack_payload=_pack_request(context, target_override=random_target),
            target_override=random_target,
            preamble="This is a random historical target control.",
        )
    except Exception:
        return generate_heuristic_bridge(
            context,
            method_name="random_target_control",
            target_override=random_target,
        )


GENERATOR_REGISTRY = {
    "orchestrator": generate_with_orchestrator,
    "single_shot_llm": generate_single_shot_llm,
    "cue_retrieval_generation": generate_cue_retrieval_generation,
    "retrieval_summary_direct": generate_retrieval_summary_direct,
    "heuristic_bridge": generate_heuristic_bridge,
    "pack_query_baseline": generate_pack_query_baseline,
    "random_target_control": generate_random_target_control,
}


def run_generation_method(method_name: str, context: GenerationContext) -> Tuple[List[GeneratedHypothesis], Dict[str, Any]]:
    if method_name not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown generation method: {method_name}")
    return GENERATOR_REGISTRY[method_name](context)

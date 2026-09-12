from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

from novelty_app.agents.observability import langchain_config_with_observability, observe_current

from .idea_fingerprint import fingerprint_text


FIELD_WEIGHTS = {
    "disease": 0.22,
    "material": 0.20,
    "payload": 0.14,
    "targeting": 0.14,
    "mechanism": 0.12,
    "model": 0.08,
    "route": 0.05,
    "outcome": 0.05,
}

IDEA_SCORE_FIELDS = (
    "importance",
    "novelty",
    "plausibility",
    "feasibility",
    "evaluability",
    "likely_impact",
)

SYSTEM_IDEA_SCORER = (
    "You are a strict AI-for-science evaluator reviewing nanomedicine research ideas. "
    "Score each hypothesis on a 1-5 integer scale for importance, novelty, plausibility, "
    "feasibility, evaluability, and likely impact. Use the evidence pack and audit context when provided. "
    "Be conservative and discriminate between speculative but weak ideas and grounded, actionable ones. "
    "Novelty should be judged relative to the provided evidence context, not to all science ever published. "
    "Feasibility should assume a capable academic lab with a 6-12 month horizon. "
    "Evaluability means the idea has clear experiments and measurable readouts. "
    "Return only structured output matching the schema."
)


class CriterionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str = ""


class HypothesisIdeaScore(BaseModel):
    hypothesis_id: str
    importance: CriterionScore
    novelty: CriterionScore
    plausibility: CriterionScore
    feasibility: CriterionScore
    evaluability: CriterionScore
    likely_impact: CriterionScore
    average_score: float = Field(ge=1.0, le=5.0)
    summary: str = ""


class HypothesisIdeaScoresOut(BaseModel):
    scored_hypotheses: List[HypothesisIdeaScore] = Field(default_factory=list)


def _field_overlap(query_terms: List[str], candidate_terms: List[str]) -> float:
    if not query_terms:
        return 0.0
    q = set(query_terms)
    c = set(candidate_terms)
    return len(q & c) / float(max(1, len(q)))


def _clamp_score(value: float) -> int:
    return max(1, min(5, int(round(value))))


def _score_summary(score_card: Dict[str, Any]) -> str:
    avg = float(score_card.get("average_score", 0.0) or 0.0)
    if avg >= 4.2:
        verdict = "strong overall"
    elif avg >= 3.2:
        verdict = "moderate overall"
    else:
        verdict = "weak overall"
    return f"{verdict}; novelty={score_card['novelty']['score']}/5, plausibility={score_card['plausibility']['score']}/5."


def _compact_papers(evidence_pack: Dict[str, Any] | None, limit: int = 10) -> List[Dict[str, Any]]:
    papers = list((evidence_pack or {}).get("papers") or [])
    compact: List[Dict[str, Any]] = []
    for paper in papers[:limit]:
        compact.append(
            {
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title", ""),
                "year": paper.get("publication_year", paper.get("year")),
                "abstract": str(paper.get("abstract", ""))[:500],
            }
        )
    return compact


def _heuristic_score_single(
    hypothesis: Dict[str, Any],
    *,
    evidence_pack: Dict[str, Any] | None = None,
    grounding_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    title = str(hypothesis.get("title") or "")
    text = str(hypothesis.get("text") or hypothesis.get("mechanistic_rationale") or "")
    support_citations = list(hypothesis.get("support_citations") or hypothesis.get("citations") or [])
    grounding_summary = dict(grounding_summary or hypothesis.get("grounding_summary") or {})
    fingerprint = dict(hypothesis.get("idea_fingerprint") or fingerprint_text(f"{title} {text}"))

    field_count = sum(1 for field in FIELD_WEIGHTS if fingerprint.get(field))
    supported_fraction = float(grounding_summary.get("supported_claim_fraction", 0.6) or 0.0)
    n_evidence = int(
        grounding_summary.get("n_evidence_papers")
        or len((evidence_pack or {}).get("papers") or [])
        or 0
    )
    support_count = len(support_citations)
    lower = f"{title} {text}".lower()

    importance = _clamp_score(
        2.0
        + (1.0 if fingerprint.get("disease") else 0.0)
        + (0.5 if fingerprint.get("payload") else 0.0)
        + (0.5 if any(term in lower for term in ("cancer", "fibrosis", "infection", "metast")) else 0.0)
    )
    novelty = _clamp_score(
        1.5
        + (1.0 if field_count >= 3 else 0.0)
        + (1.0 if field_count >= 5 else 0.0)
        + (0.5 if fingerprint.get("targeting") and fingerprint.get("payload") else 0.0)
    )
    plausibility = _clamp_score(
        1.0
        + 2.0 * supported_fraction
        + (0.5 if support_count >= 1 else 0.0)
        + (0.5 if n_evidence >= 5 else 0.0)
    )
    feasibility = _clamp_score(
        1.0
        + (1.0 if n_evidence >= 3 else 0.0)
        + (1.0 if support_count >= 1 else 0.0)
        + (0.5 if len(text) <= 900 else 0.0)
        + (0.5 if any(term in lower for term in ("in vitro", "in vivo", "mouse", "xenograft", "knockdown", "release")) else 0.0)
    )
    evaluability = _clamp_score(
        1.0
        + (1.0 if field_count >= 3 else 0.0)
        + (1.0 if support_count >= 1 else 0.0)
        + (1.0 if any(term in lower for term in ("uptake", "release", "knockdown", "tumor", "imaging", "survival")) else 0.0)
    )
    likely_impact = _clamp_score((importance + novelty + plausibility) / 3.0)

    score_card = {
        "importance": {"score": importance, "rationale": "Heuristic estimate based on disease relevance and application scope."},
        "novelty": {"score": novelty, "rationale": "Heuristic estimate based on the number of distinct bridge facets combined."},
        "plausibility": {"score": plausibility, "rationale": "Heuristic estimate based on grounding support and evidence context."},
        "feasibility": {"score": feasibility, "rationale": "Heuristic estimate based on evidence support and apparent experimental tractability."},
        "evaluability": {"score": evaluability, "rationale": "Heuristic estimate based on whether concrete readouts appear testable."},
        "likely_impact": {"score": likely_impact, "rationale": "Heuristic estimate from importance, novelty, and plausibility."},
        "average_score": round((importance + novelty + plausibility + feasibility + evaluability + likely_impact) / 6.0, 2),
        "summary": "",
        "score_method": "heuristic_fallback",
        "judge_model": "heuristic_fallback",
    }
    score_card["summary"] = _score_summary(score_card)
    return score_card


def _heuristic_score_hypotheses(
    hypotheses: Sequence[Dict[str, Any]],
    *,
    evidence_pack: Dict[str, Any] | None = None,
    grounding_summary: Dict[str, Any] | None = None,
) -> Dict[str, Dict[str, Any]]:
    scored: Dict[str, Dict[str, Any]] = {}
    for idx, hypothesis in enumerate(hypotheses):
        hypothesis_id = str(hypothesis.get("hypothesis_id") or hypothesis.get("id") or f"hyp_{idx}")
        score_card = _heuristic_score_single(
            hypothesis,
            evidence_pack=evidence_pack,
            grounding_summary=grounding_summary,
        )
        score_card["hypothesis_id"] = hypothesis_id
        scored[hypothesis_id] = score_card
    return scored


def _llm_api_key(explicit_api_key: str | None = None) -> str | None:
    return explicit_api_key or os.getenv("OPENAI_API_KEY")


def _judge_trace_metadata(
    *,
    target: Dict[str, Any] | None,
    discovery_cue: Dict[str, Any] | None,
    n_hypotheses: int,
) -> Dict[str, Any]:
    target_payload = dict(target or {})
    return {
        "snapshot_id": target_payload.get("snapshot_id"),
        "target_type": target_payload.get("target_type"),
        "gap_id": target_payload.get("gap_id"),
        "cluster_a": target_payload.get("cluster_a"),
        "cluster_b": target_payload.get("cluster_b"),
        "n_hypotheses": n_hypotheses,
        "has_discovery_cue": bool((discovery_cue or {}).get("text")),
    }


def score_hypotheses(
    hypotheses: Sequence[Dict[str, Any]],
    *,
    evidence_pack: Dict[str, Any] | None = None,
    audit: Dict[str, Any] | None = None,
    explanation: Dict[str, Any] | None = None,
    target: Dict[str, Any] | None = None,
    discovery_cue: Dict[str, Any] | None = None,
    openai_api_key: str | None = None,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    if not hypotheses:
        return {}

    api_key = _llm_api_key(openai_api_key)
    model = model_name or os.getenv("OPENAI_EVAL_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07")
    if ChatOpenAI is None or not api_key:
        return _heuristic_score_hypotheses(hypotheses, evidence_pack=evidence_pack, grounding_summary=audit)

    compact_hypotheses: List[Dict[str, Any]] = []
    for idx, hypothesis in enumerate(hypotheses):
        compact_hypotheses.append(
            {
                "hypothesis_id": str(hypothesis.get("hypothesis_id") or hypothesis.get("id") or f"hyp_{idx}"),
                "title": hypothesis.get("title"),
                "text": hypothesis.get("text") or hypothesis.get("mechanistic_rationale"),
                "support_citations": list(hypothesis.get("support_citations") or hypothesis.get("citations") or []),
                "grounding_summary": hypothesis.get("grounding_summary") or {},
                "idea_fingerprint": hypothesis.get("idea_fingerprint") or {},
            }
        )

    prompt = f"""
TARGET:
{json.dumps(target or {}, ensure_ascii=False)}

DISCOVERY_CUE:
{json.dumps(discovery_cue or {}, ensure_ascii=False)}

AUDIT_CONTEXT:
{json.dumps(audit or {}, ensure_ascii=False)}

EXPLANATION_CONTEXT:
{json.dumps(explanation or {}, ensure_ascii=False)[:2500]}

EVIDENCE_PACK:
{json.dumps(_compact_papers(evidence_pack), ensure_ascii=False)}

HYPOTHESES:
{json.dumps(compact_hypotheses, ensure_ascii=False)}

Score every hypothesis from 1-5 for:
- importance
- novelty
- plausibility
- feasibility
- evaluability
- likely_impact

Return a brief rationale per criterion and an average_score for each hypothesis.
"""

    try:
        options = {"reasoning_effort": reasoning_effort} if reasoning_effort else {"temperature": 0.0}
        llm = ChatOpenAI(model=model, api_key=api_key, **options)
        structured = llm.with_structured_output(HypothesisIdeaScoresOut, method="function_calling")
        messages = [
            {"role": "system", "content": SYSTEM_IDEA_SCORER},
            {"role": "user", "content": prompt},
        ]
        trace_metadata = _judge_trace_metadata(
            target=target,
            discovery_cue=discovery_cue,
            n_hypotheses=len(compact_hypotheses),
        )
        with observe_current(
            name="judge_hypotheses",
            as_type="evaluator",
            input_payload={
                "target": target or {},
                "n_hypotheses": len(compact_hypotheses),
                "evidence_size": len((evidence_pack or {}).get("papers") or []),
            },
            metadata=trace_metadata,
            model=model,
        ) as observation:
            out = structured.invoke(
                messages,
                config=langchain_config_with_observability(
                    session_id=str((target or {}).get("snapshot_id") or "") or None,
                    tags=["judge", str((target or {}).get("target_type") or "unknown")],
                    trace_name="judge_hypotheses",
                    metadata=trace_metadata,
                ),
            )
            observation.update(
                output={
                    "n_scored_hypotheses": len(out.scored_hypotheses),
                    "judge_model": model,
                }
            )
        scored: Dict[str, Dict[str, Any]] = {}
        for item in out.scored_hypotheses:
            payload = item.model_dump()
            payload["score_method"] = "llm_judge"
            payload["judge_model"] = model
            scored[item.hypothesis_id] = payload
        if len(scored) == len(compact_hypotheses):
            return scored
    except Exception:
        pass

    heuristic = _heuristic_score_hypotheses(hypotheses, evidence_pack=evidence_pack, grounding_summary=audit)
    for hypothesis_id, score_card in heuristic.items():
        score_card.setdefault("score_method", "heuristic_fallback")
        score_card.setdefault("judge_model", "heuristic_fallback")
    return heuristic


def judge_candidate_match(
    fingerprint: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    weights: Tuple[float, float, float] = (0.55, 0.25, 0.20),
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    if len(weights) != 3 or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Three nonnegative matching weights with positive sum are required")
    weights = tuple(w / sum(weights) for w in weights)
    limits = {"strong_rank": 0.80, "strong_overlap": 0.45, "strong_combined": 0.70,
              "partial_rank": 0.58, "partial_overlap": 0.22, "partial_combined": 0.50,
              "background_rank": 0.38, "background_overlap": 0.15}
    if thresholds:
        if set(thresholds) - set(limits) or any(not 0 <= v <= 1 for v in thresholds.values()):
            raise ValueError("Unknown or out-of-range matching threshold")
        limits.update(thresholds)
    candidate_fp = fingerprint_text(
        f"{candidate.get('title', '')} {candidate.get('abstract', candidate.get('text', ''))}"
    )
    field_scores: Dict[str, float] = {}
    overlap = 0.0
    for field, weight in FIELD_WEIGHTS.items():
        score = _field_overlap(fingerprint.get(field, []), candidate_fp.get(field, []))
        field_scores[field] = score
        overlap += weight * score

    rerank = float(candidate.get("reranker_score", 0.0) or 0.0)
    emb = float(candidate.get("embedding_score", 0.0) or 0.0)
    emb_norm = (emb + 1.0) / 2.0
    combined = weights[0] * rerank + weights[1] * overlap + weights[2] * emb_norm

    if rerank >= limits["strong_rank"] and overlap >= limits["strong_overlap"] and combined >= limits["strong_combined"]:
        label = "strong_match"
    elif rerank >= limits["partial_rank"] and overlap >= limits["partial_overlap"] and combined >= limits["partial_combined"]:
        label = "partial_match"
    elif rerank >= limits["background_rank"] or overlap >= limits["background_overlap"]:
        label = "background_only"
    else:
        label = "no_match"

    return {
        "label": label,
        "combined_score": combined,
        "field_overlap": overlap,
        "field_scores": field_scores,
        "candidate_fingerprint": candidate_fp,
    }


def normalize_cue_score(score: Optional[float]) -> float:
    if score is None:
        return 1.0
    normalized = (float(score) + 1.0) / 2.5
    return max(0.0, min(1.0, normalized))


def classify_recovery_match(
    *,
    historical_best: Dict[str, Any],
    gold_rank: Optional[int],
    best_future_neighbor: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    historical_label = historical_best.get("judge", {}).get("label", "no_match")
    future_neighbor_label = best_future_neighbor.get("judge", {}).get("label", "no_match")

    if historical_label == "strong_match":
        recovery_label = "historical_confound"
    elif gold_rank is not None and gold_rank <= 10:
        recovery_label = "gold_recovered"
    elif future_neighbor_label in {"strong_match", "partial_match", "background_only"}:
        recovery_label = "future_neighbor_only"
    else:
        recovery_label = "not_recovered"

    return recovery_label, {
        "historical_label": historical_label,
        "future_neighbor_label": future_neighbor_label,
    }

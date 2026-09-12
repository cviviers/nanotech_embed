"""Read-only domain snapshot preparation and strict later-literature retrieval."""
from __future__ import annotations

import sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np

from novelty_app.agents.knowledge_store import KnowledgeStore
from .controlled_study import digest, freeze_papers, read_json

DOMAIN_FOLDERS = {
    "antimicrobials": "retrospective_eval_antimicrobials_full_20260328",
    "biosensing": "retrospective_eval_biosensing_full_20260329",
    "payload": "retrospective_eval_payload_full_20260329",
    "vaccine": "retrospective_eval_vaccine_full_20260329",
}


class IneligibleTask(ValueError):
    """A historical task lacks the source evidence required by the frozen protocol."""


class ReadOnlyStore(KnowledgeStore):
    def __init__(self, path: Path, qwen_url="http://localhost:8000"):
        self.db_path = path.resolve(strict=True)
        self._cue_similarity_cache = {}
        self.qwen_url = qwen_url

    def _embed_texts_with_qwen(self, texts):
        from .qwen_client import QwenClient
        return QwenClient(self.qwen_url).embed(list(texts), normalize=True)

    def _open_connection(self):
        conn = sqlite3.connect(self.db_path.as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        return conn

    def _connect(self):
        return self._open_connection()


def domain_sources(root: Path, qwen_url="http://localhost:8000"):
    for domain, folder in DOMAIN_FOLDERS.items():
        directory = root / folder
        bundles = sorted(directory.glob("*_assessment_bundle_v1.json"))
        if len(bundles) != 1:
            raise ValueError(f"Expected one source bundle in {directory}, found {len(bundles)}")
        manifest = read_json(bundles[0])["run_manifest"]
        yield domain, ReadOnlyStore(directory / "novelty_agent_knowledge.sqlite", qwen_url), manifest


def historical_pairs(store: ReadOnlyStore, snapshot_id: str):
    # LIMIT -1 enumerates every stored gap, without the prospective runner's arbitrary-pair fallback.
    pairs = {}
    for gap in store.top_gaps(snapshot_id, k=-1):
        clusters = sorted({int(c) for c in gap.get("cluster_ids", []) if c is not None and int(c) >= 0})
        for pair in combinations(clusters, 2):
            pairs.setdefault(pair, []).append(gap["gap_id"])
    return [(a, b, sorted(gaps)) for (a, b), gaps in sorted(pairs.items())]


def prepare_task(domain, store, manifest, a, b, gaps):
    sid = manifest["snapshot_id"]
    cue = manifest.get("discovery_cue") or manifest["config"]["discovery_cue"]
    config = manifest["config"]
    target = {"cluster_a": a, "cluster_b": b}
    task_id = digest([domain, sid, cue, a, b])[:24]
    request = {"snapshot_id": sid, "target_type": "cluster_pair", **target,
               "profile": "focused_eval", "exemplars": 8, "boundary": 8, "diverse": 0,
               "counter_queries": [], "discovery_cue": cue,
               "cue_source_snapshot_id": config.get("cue_source_snapshot_id") or sid,
               "cue_similarity_top_k": config.get("cue_similarity_top_k", 50),
               "cue_similarity_sample_n": config.get("cue_similarity_sample_n", 6),
               "cue_similarity_seed": task_id}
    pack = store.build_evidence_pack(request)
    papers = freeze_papers(pack["papers"])
    # Cluster IDs from a cue-source snapshot are not memberships in the task snapshot.
    with store._connect() as conn:
        memberships = {r["paper_id"]: r["cluster_id"] for r in conn.execute(
            "SELECT paper_id, cluster_id FROM papers WHERE snapshot_id=?", (sid,))}
    for paper in papers:
        paper["cluster_id"] = memberships.get(paper["paper_id"])
        if paper["publication_year"] is None or int(paper["publication_year"]) > int(manifest["cutoff_date"][:4]):
            raise IneligibleTask("Missing publication year or post-cutoff source in historical pack")
    for cluster in (a, b):
        if not any(p["cluster_id"] == cluster and p["title"].strip() and p["abstract"].strip() for p in papers):
            raise IneligibleTask(f"No title/abstract source for community {cluster}")
    task = {"task_id": task_id, "domain": domain, "snapshot_id": sid, "cue": cue,
            "target": target, "source_gaps": gaps, "papers": papers,
            "pack_hash": digest(papers), "retrieval_request": request,
            "cutoff_date": manifest["cutoff_date"]}
    task["task_hash"] = digest(task)
    return task


def fused_candidates(lexical: list[int], semantic: list[int], k=60) -> list[tuple[int, float]]:
    scores = {}
    for ranking in (lexical, semantic):
        for rank, index in enumerate(ranking, 1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def retrieve_later(candidate: dict, corpus, qwen) -> list[dict]:
    from .idea_fingerprint import fingerprint_text
    query = candidate["title"] + "\n" + candidate["text"]
    fingerprint = fingerprint_text(query)
    terms = {str(t).lower() for key, values in fingerprint.items()
             if isinstance(values, list) for t in values if isinstance(t, str)}
    if not terms:
        terms = {t.lower() for t in query.split() if len(t) > 3}
    lexical_scores = np.array([sum(t in text for t in terms) for text in corpus.lower_texts])
    embedding = np.asarray(qwen.embed([query], instruction="Retrieve scientific papers that describe the same concrete research idea.", normalize=True)[0])
    if embedding.shape != (corpus.normalized_embeddings.shape[1],) or not np.isfinite(embedding).all():
        raise ValueError("Invalid retrieval embedding")
    semantic_scores = corpus.normalized_embeddings @ embedding
    lexical = [int(i) for i in np.argsort(-lexical_scores, kind="stable")[:60] if lexical_scores[i] > 0]
    semantic = [int(i) for i in np.argsort(-semantic_scores, kind="stable")[:120]]
    fused = fused_candidates(lexical, semantic)[:64]
    if not fused:
        return []
    documents = [corpus.texts[i] for i, _ in fused]
    ranked = qwen.rank(query=query, documents=documents, top_k=None, require_reranker=True)
    if len(ranked) != len(documents) or {int(r["index"]) for r in ranked} != set(range(len(documents))):
        raise ValueError("Reranker returned incomplete or duplicate indices")
    result = []
    for r in ranked:
        i, fusion = fused[int(r["index"])]
        score = r.get("reranker_score", r.get("score"))
        if score is None or not np.isfinite(float(score)):
            raise ValueError("Missing reranker score")
        row = corpus.df.iloc[i]
        result.append({"paper_id": corpus.paper_ids[i], "title": str(row.get("title") or ""),
                       "abstract": str(row.get("abstract") or ""), "text": corpus.texts[i],
                       "lexical_score": int(lexical_scores[i]), "semantic_score": float(semantic_scores[i]),
                       "fusion_score": fusion, "reranker_score": float(score)})
    result.sort(key=lambda r: (-r["reranker_score"], r["paper_id"]))
    return [{**r, "rank": rank} for rank, r in enumerate(result, 1)]

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

try:
    from agents.backend_client import BackendClient
    from agents.schemas import DiscoveryCue
except Exception:  # pragma: no cover
    from novelty_app.agents.backend_client import BackendClient
    from novelty_app.agents.schemas import DiscoveryCue


class TopGapsArgs(BaseModel):
    snapshot_id: str | None = Field(default=None, description="Snapshot id. If omitted, latest snapshot is used.")
    k: int = Field(default=25, ge=1, le=200, description="Maximum number of gap candidates to return.")


class ListClustersArgs(BaseModel):
    snapshot_id: str | None = Field(default=None, description="Snapshot id. If omitted, latest snapshot is used.")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of clusters to return.")
    sort: str = Field(default="size_desc", description="Sort mode: size_desc, size_asc, cluster_id_asc, cluster_id_desc.")


class PapersBatchArgs(BaseModel):
    snapshot_id: str = Field(description="Snapshot id for the query.")
    paper_ids: List[str] = Field(description="List of paper ids to fetch.")
    fields: List[str] = Field(default_factory=list, description="Optional field subset to return.")


class EvidencePackArgs(BaseModel):
    snapshot_id: str | None = Field(default=None, description="Snapshot id. If omitted, latest snapshot is used.")
    target_type: str = Field(description="Target type: 'gap', 'cluster_pair', or cue-only 'cue'.")
    gap_id: str | None = Field(default=None, description="Gap id when target_type='gap'.")
    cluster_a: int | None = Field(default=None, description="Cluster A id when target_type='cluster_pair'.")
    cluster_b: int | None = Field(default=None, description="Cluster B id when target_type='cluster_pair'.")
    required_paper_ids: List[str] = Field(
        default_factory=list,
        description="Optional paper ids to force-include in the evidence pack.",
    )
    required_paper_source_snapshot_id: str | None = Field(
        default=None,
        description="Optional published snapshot from which required paper ids are fetched.",
    )
    exemplars: int = Field(default=25, ge=0, le=200)
    boundary: int = Field(default=25, ge=0, le=200)
    diverse: int = Field(default=25, ge=0, le=200)
    counter_queries: List[str] = Field(
        default_factory=list,
        description="Keyword queries used to patch missing evidence; matched server-side over titles/abstracts.",
    )
    discovery_cue: DiscoveryCue | None = Field(
        default=None,
        description="Optional steering cue describing the desired research direction. This is not treated as evidence.",
    )
    cue_source_snapshot_id: str | None = Field(
        default=None,
        description="Snapshot id used for cue-semantic similarity retrieval (required when discovery cue is active).",
    )
    cue_similarity_top_k: int = Field(default=50, ge=1, le=5000)
    cue_similarity_sample_n: int = Field(default=6, ge=0, le=500)
    cue_similarity_seed: str | int | None = Field(
        default=None,
        description="Optional deterministic seed used when sampling cue-similar papers from the top-k set.",
    )


class StoreArtifactArgs(BaseModel):
    snapshot_id: str | None = Field(default=None, description="Optional snapshot id to associate with this artifact.")
    kind: str = Field(description="Artifact kind, e.g., research_brief.")
    target: Dict[str, Any] = Field(default_factory=dict, description="Target descriptor (gap, cluster pair, etc.).")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Artifact body.")


def make_backend_tools(backend: BackendClient) -> List[Any]:
    """Create LangChain structured tools over the novelty agent backend."""
    try:
        from langchain_core.tools import StructuredTool
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "langchain-core is required to build structured tools. "
            "Install langchain-core/langchain/langgraph in your agent environment."
        ) from exc

    def _top_gaps(snapshot_id: str | None = None, k: int = 25) -> Dict[str, Any]:
        return backend.top_gaps(snapshot_id=snapshot_id, k=k)

    def _list_clusters(snapshot_id: str | None = None, limit: int = 100, sort: str = "size_desc") -> Dict[str, Any]:
        return backend.list_clusters(snapshot_id=snapshot_id, limit=limit, sort=sort)

    def _papers_batch(snapshot_id: str, paper_ids: List[str], fields: List[str] | None = None) -> Dict[str, Any]:
        return backend.papers_batch(snapshot_id=snapshot_id, paper_ids=paper_ids, fields=fields or None)

    def _evidence_pack(
        snapshot_id: str | None = None,
        target_type: str = "gap",
        gap_id: str | None = None,
        cluster_a: int | None = None,
        cluster_b: int | None = None,
        required_paper_ids: List[str] | None = None,
        required_paper_source_snapshot_id: str | None = None,
        exemplars: int = 25,
        boundary: int = 25,
        diverse: int = 25,
        counter_queries: List[str] | None = None,
        discovery_cue: DiscoveryCue | None = None,
        cue_source_snapshot_id: str | None = None,
        cue_similarity_top_k: int = 50,
        cue_similarity_sample_n: int = 6,
        cue_similarity_seed: str | int | None = None,
    ) -> Dict[str, Any]:
        return backend.evidence_pack(
            {
                "snapshot_id": snapshot_id,
                "target_type": target_type,
                "gap_id": gap_id,
                "cluster_a": cluster_a,
                "cluster_b": cluster_b,
                "required_paper_ids": required_paper_ids or [],
                "required_paper_source_snapshot_id": required_paper_source_snapshot_id,
                "exemplars": exemplars,
                "boundary": boundary,
                "diverse": diverse,
                "counter_queries": counter_queries or [],
                "discovery_cue": discovery_cue.model_dump() if discovery_cue is not None else None,
                "cue_source_snapshot_id": cue_source_snapshot_id,
                "cue_similarity_top_k": cue_similarity_top_k,
                "cue_similarity_sample_n": cue_similarity_sample_n,
                "cue_similarity_seed": cue_similarity_seed,
            }
        )

    def _store_artifact(
        kind: str,
        target: Dict[str, Any],
        payload: Dict[str, Any],
        snapshot_id: str | None = None,
    ) -> Dict[str, Any]:
        if snapshot_id is not None:
            target = dict(target or {})
            target.setdefault("snapshot_id", snapshot_id)
        return backend.store_artifact(kind=kind, target=target, payload=payload)

    return [
        StructuredTool.from_function(
            name="get_top_gap_candidates",
            description="List top novelty/gap candidates from the current snapshot.",
            func=_top_gaps,
            args_schema=TopGapsArgs,
        ),
        StructuredTool.from_function(
            name="list_clusters",
            description="List clusters in the snapshot with sizes and metadata.",
            func=_list_clusters,
            args_schema=ListClustersArgs,
        ),
        StructuredTool.from_function(
            name="build_evidence_pack",
            description="Build an evidence pack for a gap or cluster pair, optionally force-including paper ids from another snapshot.",
            func=_evidence_pack,
            args_schema=EvidencePackArgs,
        ),
        StructuredTool.from_function(
            name="fetch_papers_batch",
            description="Fetch full or partial paper records by paper_id.",
            func=_papers_batch,
            args_schema=PapersBatchArgs,
        ),
        StructuredTool.from_function(
            name="store_artifact",
            description="Persist an agent-generated artifact (e.g., research brief) with target metadata.",
            func=_store_artifact,
            args_schema=StoreArtifactArgs,
        ),
    ]

from __future__ import annotations

import json
import os
import hashlib
import random
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from discovery_cue import discovery_cue_query_terms, discovery_cue_to_dict, normalize_discovery_cue, score_record_against_cue
except Exception:  # pragma: no cover
    from novelty_app.discovery_cue import (
        discovery_cue_query_terms,
        discovery_cue_to_dict,
        normalize_discovery_cue,
        score_record_against_cue,
    )


_REQUIRED_TABLES = {
    "artifacts",
    "clusters",
    "evaluation_matches",
    "evaluation_runs",
    "gap_papers",
    "gaps",
    "llm_analyses",
    "papers",
    "snapshots",
}


def default_db_path() -> Path:
    configured = os.getenv("NOVELTY_AGENT_DB")
    if configured:
        return Path(configured).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / "data" / "novelty_agent_knowledge.sqlite").resolve()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass

    return str(obj)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _json_loads(text: Optional[str], default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_text(value: Any) -> Optional[str]:
    """Coerce arbitrary payload values into SQLite-safe text."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    # Preserve numeric/bool values as strings in text columns.
    if isinstance(value, (int, float, bool)):
        return str(value)
    # Container/array-like payloads are serialized as JSON for traceability.
    if isinstance(value, (dict, list, tuple, set)):
        return _json_dumps(value)
    if hasattr(value, "tolist"):
        try:
            return _json_dumps(value.tolist())
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    try:
        return str(value)
    except Exception:
        return _json_dumps(value)


def _normalize_paper_ids(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


_PAPER_ID_ALIAS_PREFIXES = ("pmid", "id", "paper_id", "doi")


def _escape_sql_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _split_paper_id_prefix(value: str) -> Tuple[Optional[str], Optional[str]]:
    prefix, sep, raw_value = str(value or "").partition(":")
    if not sep:
        return None, None
    normalized_prefix = prefix.strip().lower()
    normalized_raw_value = raw_value.strip()
    if not normalized_prefix or not normalized_raw_value:
        return None, None
    return normalized_prefix, normalized_raw_value


def _to_embedding_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = _json_loads(value, None)
        if parsed is None:
            return None
        value = parsed
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            return None
    if not isinstance(value, (list, tuple)):
        return None
    out: List[float] = []
    for x in value:
        fx = _to_float(x)
        if fx is None:
            return None
        out.append(fx)
    if not out:
        return None
    return out
    try:
        return float(value)
    except Exception:
        return None


class KnowledgeStore:
    """SQLite-backed knowledge store for agent-accessible novelty analysis artifacts."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._cue_similarity_cache: Dict[str, Dict[str, Any]] = {}

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _connect(self) -> sqlite3.Connection:
        conn = self._open_connection()
        self._ensure_schema(conn)
        return conn

    def _init_schema(self) -> None:
        with self._open_connection() as conn:
            self._ensure_schema(conn)

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        existing = {
            str(r[0])
            for r in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            if r[0]
        }
        if not _REQUIRED_TABLES.issubset(existing):
            self._apply_schema(cur)
            conn.commit()
            return
        self._ensure_runtime_columns(cur)
        conn.commit()

    def _apply_schema(self, cur: sqlite3.Cursor) -> None:
        try:
            cur.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.OperationalError:
            # WAL can fail on synced folders (e.g., OneDrive); fallback to default journal mode.
            pass
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS papers (
                snapshot_id TEXT NOT NULL,
                paper_id TEXT NOT NULL,
                row_index INTEGER,
                title TEXT,
                abstract TEXT,
                publication_year INTEGER,
                doi TEXT,
                journal TEXT,
                cluster_id INTEGER,
                gap_score REAL,
                gap_region INTEGER,
                embedding_json TEXT,
                embedding_dim INTEGER,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (snapshot_id, paper_id)
            );
            CREATE INDEX IF NOT EXISTS idx_papers_snapshot_cluster ON papers(snapshot_id, cluster_id);
            CREATE INDEX IF NOT EXISTS idx_papers_snapshot_gap_region ON papers(snapshot_id, gap_region);
            CREATE INDEX IF NOT EXISTS idx_papers_snapshot_gap_score ON papers(snapshot_id, gap_score);

            CREATE TABLE IF NOT EXISTS clusters (
                snapshot_id TEXT NOT NULL,
                cluster_id INTEGER NOT NULL,
                size INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (snapshot_id, cluster_id)
            );
            CREATE INDEX IF NOT EXISTS idx_clusters_snapshot_size ON clusters(snapshot_id, size);

            CREATE TABLE IF NOT EXISTS gaps (
                snapshot_id TEXT NOT NULL,
                gap_id TEXT NOT NULL,
                region_index INTEGER,
                size INTEGER NOT NULL,
                avg_gap_score REAL,
                max_gap_score REAL,
                density_z REAL,
                cluster_ids_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (snapshot_id, gap_id)
            );
            CREATE INDEX IF NOT EXISTS idx_gaps_snapshot_avg ON gaps(snapshot_id, avg_gap_score);

            CREATE TABLE IF NOT EXISTS gap_papers (
                snapshot_id TEXT NOT NULL,
                gap_id TEXT NOT NULL,
                paper_id TEXT NOT NULL,
                rank_idx INTEGER,
                gap_score REAL,
                PRIMARY KEY (snapshot_id, gap_id, paper_id)
            );
            CREATE INDEX IF NOT EXISTS idx_gap_papers_snapshot_gap ON gap_papers(snapshot_id, gap_id, rank_idx);

            CREATE TABLE IF NOT EXISTS llm_analyses (
                analysis_id TEXT PRIMARY KEY,
                snapshot_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                target_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_llm_analyses_snapshot ON llm_analyses(snapshot_id, created_at);

            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                snapshot_id TEXT,
                kind TEXT NOT NULL,
                created_at TEXT NOT NULL,
                target_json TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_artifacts_snapshot_created ON artifacts(snapshot_id, created_at);

            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id TEXT PRIMARY KEY,
                snapshot_id TEXT,
                created_at TEXT NOT NULL,
                cutoff_date TEXT,
                future_window_start TEXT,
                future_window_end TEXT,
                method_names_json TEXT NOT NULL,
                config_json TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                observability_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_evaluation_runs_snapshot_created ON evaluation_runs(snapshot_id, created_at);

            CREATE TABLE IF NOT EXISTS evaluation_matches (
                match_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                snapshot_id TEXT,
                created_at TEXT NOT NULL,
                target_id TEXT NOT NULL,
                target_type TEXT NOT NULL,
                method_name TEXT NOT NULL,
                seed INTEGER,
                hypothesis_id TEXT NOT NULL,
                classification TEXT NOT NULL,
                historical_label TEXT,
                future_label TEXT,
                first_future_year INTEGER,
                historical_best_paper_id TEXT,
                future_best_paper_id TEXT,
                support_citations_json TEXT NOT NULL,
                hypothesis_json TEXT NOT NULL,
                idea_scores_json TEXT NOT NULL DEFAULT '{}',
                fingerprint_json TEXT NOT NULL,
                historical_match_json TEXT NOT NULL,
                future_match_json TEXT NOT NULL,
                trace_ref_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_evaluation_matches_run_method ON evaluation_matches(run_id, method_name, seed);
            CREATE INDEX IF NOT EXISTS idx_evaluation_matches_snapshot_target ON evaluation_matches(snapshot_id, target_id, classification);
            """
        )
        self._ensure_runtime_columns(cur)

    def _ensure_runtime_columns(self, cur: sqlite3.Cursor) -> None:
        self._ensure_table_columns(
            cur,
            "papers",
            {
                "embedding_json": "TEXT",
                "embedding_dim": "INTEGER",
            },
        )
        self._ensure_table_columns(
            cur,
            "evaluation_matches",
            {
                "idea_scores_json": "TEXT",
                "trace_ref_json": "TEXT",
                "recovery_label": "TEXT",
                "future_neighbor_label": "TEXT",
                "gold_future_paper_id": "TEXT",
                "gold_future_title": "TEXT",
                "gold_future_year": "INTEGER",
                "assigned_target_id": "TEXT",
                "assigned_target_score": "REAL",
                "gold_rank": "INTEGER",
                "gold_reciprocal_rank": "REAL",
                "gold_hit_at_1": "INTEGER",
                "gold_hit_at_5": "INTEGER",
                "gold_hit_at_10": "INTEGER",
                "cue_score": "REAL",
                "cue_weighted_rr": "REAL",
                "best_future_neighbor_paper_id": "TEXT",
                "best_historical_confounder_id": "TEXT",
                "evidence_pack_summary_json": "TEXT",
                "historical_candidates_json": "TEXT",
                "future_candidates_json": "TEXT",
            },
        )
        self._ensure_table_columns(
            cur,
            "evaluation_runs",
            {
                "observability_json": "TEXT",
            },
        )

    def _ensure_table_columns(self, cur: sqlite3.Cursor, table: str, columns: Dict[str, str]) -> None:
        existing = {str(r[1]) for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, col_type in columns.items():
            if name in existing:
                continue
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")

    def _delete_snapshot(self, conn: sqlite3.Connection, snapshot_id: str) -> None:
        cur = conn.cursor()
        for table in ("papers", "clusters", "gaps", "gap_papers", "llm_analyses"):
            cur.execute(f"DELETE FROM {table} WHERE snapshot_id = ?", (snapshot_id,))
        cur.execute("DELETE FROM snapshots WHERE snapshot_id = ?", (snapshot_id,))

    def publish_snapshot(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        snapshot_id = str(payload.get("snapshot_id") or uuid.uuid4())
        created_at = str(payload.get("created_at") or _utc_now_iso())
        metadata = payload.get("metadata") or {}
        papers = payload.get("papers") or []
        clusters = payload.get("clusters") or []
        gaps = payload.get("gaps") or []
        gap_papers = payload.get("gap_papers") or []
        llm_analyses = payload.get("llm_analyses") or []

        with self._connect() as conn:
            self._delete_snapshot(conn, snapshot_id)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO snapshots(snapshot_id, created_at, metadata_json) VALUES (?, ?, ?)",
                (snapshot_id, created_at, _json_dumps(metadata)),
            )

            for p in papers:
                paper_id = str(p.get("paper_id") or "")
                if not paper_id:
                    continue
                raw = p.get("raw") if isinstance(p, dict) else None
                if raw is None:
                    raw = dict(p)
                    if isinstance(raw, dict):
                        raw.pop("embedding", None)
                        raw.pop("umap_2d", None)
                embedding = _to_embedding_list(p.get("embedding")) if isinstance(p, dict) else None
                embedding_json = _json_dumps(embedding) if embedding is not None else None
                embedding_dim = len(embedding) if embedding is not None else None
                publication_year = _to_int(p.get("publication_year"))
                if publication_year is None:
                    publication_year = _to_int(p.get("year"))
                paper_params = (
                    snapshot_id,
                    paper_id,
                    _to_int(p.get("row_index")),
                    _to_text(p.get("title")),
                    _to_text(p.get("abstract")),
                    publication_year,
                    _to_text(p.get("doi")),
                    _to_text(p.get("journal")),
                    _to_int(p.get("cluster_id")),
                    _to_float(p.get("gap_score")),
                    _to_int(p.get("gap_region")),
                    embedding_json,
                    embedding_dim,
                    _json_dumps(raw),
                )
                try:
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO papers(
                            snapshot_id, paper_id, row_index, title, abstract, publication_year, doi, journal,
                            cluster_id, gap_score, gap_region, embedding_json, embedding_dim, raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        paper_params,
                    )
                except sqlite3.Error as exc:
                    field_debug = {
                        "paper_id": paper_id,
                        "types": {
                            "row_index": type(p.get("row_index")).__name__,
                            "title": type(p.get("title")).__name__,
                            "abstract": type(p.get("abstract")).__name__,
                            "publication_year": type(p.get("publication_year")).__name__,
                            "year": type(p.get("year")).__name__,
                            "doi": type(p.get("doi")).__name__,
                            "journal": type(p.get("journal")).__name__,
                            "cluster_id": type(p.get("cluster_id")).__name__,
                            "gap_score": type(p.get("gap_score")).__name__,
                            "gap_region": type(p.get("gap_region")).__name__,
                            "embedding": type(p.get("embedding")).__name__,
                        },
                    }
                    raise sqlite3.Error(f"{exc}; paper_debug={field_debug}") from exc

            for c in clusters:
                cid = _to_int(c.get("cluster_id"))
                if cid is None:
                    continue
                cur.execute(
                    """
                    INSERT OR REPLACE INTO clusters(snapshot_id, cluster_id, size, metadata_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        cid,
                        int(c.get("size") or 0),
                        _json_dumps(c.get("metadata") or {}),
                    ),
                )

            for g in gaps:
                gid = str(g.get("gap_id") or "")
                if not gid:
                    continue
                cur.execute(
                    """
                    INSERT OR REPLACE INTO gaps(
                        snapshot_id, gap_id, region_index, size, avg_gap_score, max_gap_score, density_z,
                        cluster_ids_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        gid,
                        _to_int(g.get("region_index")),
                        int(g.get("size") or 0),
                        _to_float(g.get("avg_gap_score")),
                        _to_float(g.get("max_gap_score")),
                        _to_float(g.get("density_z")),
                        _json_dumps(g.get("cluster_ids") or []),
                        _json_dumps(g.get("metadata") or {}),
                    ),
                )

            for gp in gap_papers:
                gid = str(gp.get("gap_id") or "")
                pid = str(gp.get("paper_id") or "")
                if not gid or not pid:
                    continue
                cur.execute(
                    """
                    INSERT OR REPLACE INTO gap_papers(snapshot_id, gap_id, paper_id, rank_idx, gap_score)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        gid,
                        pid,
                        _to_int(gp.get("rank") if gp.get("rank") is not None else gp.get("rank_idx")),
                        _to_float(gp.get("gap_score")),
                    ),
                )

            for a in llm_analyses:
                analysis_id = str(a.get("analysis_id") or uuid.uuid4())
                cur.execute(
                    """
                    INSERT OR REPLACE INTO llm_analyses(
                        analysis_id, snapshot_id, created_at, target_json, result_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        analysis_id,
                        snapshot_id,
                        str(a.get("created_at") or _utc_now_iso()),
                        _json_dumps(a.get("target") or {}),
                        _json_dumps(a.get("result") or {}),
                        _json_dumps(a.get("metadata") or {}),
                    ),
                )

            conn.commit()

        return {
            "snapshot_id": snapshot_id,
            "created_at": created_at,
            "counts": {
                "papers": len(papers),
                "clusters": len(clusters),
                "gaps": len(gaps),
                "gap_papers": len(gap_papers),
                "llm_analyses": len(llm_analyses),
            },
        }

    def resolve_snapshot_id(self, snapshot_id: Optional[str]) -> str:
        if snapshot_id:
            return snapshot_id
        with self._connect() as conn:
            row = conn.execute(
                "SELECT snapshot_id FROM snapshots ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            raise ValueError("No snapshots available")
        return str(row["snapshot_id"])

    def list_snapshots(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT snapshot_id, created_at, metadata_json FROM snapshots ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "snapshot_id": r["snapshot_id"],
                    "created_at": r["created_at"],
                    "metadata": _json_loads(r["metadata_json"], {}),
                }
            )
        return out

    def get_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT snapshot_id, created_at, metadata_json FROM snapshots WHERE snapshot_id = ?",
                (str(snapshot_id),),
            ).fetchone()
        if not row:
            raise ValueError(f"Snapshot not found: {snapshot_id}")
        return {
            "snapshot_id": row["snapshot_id"],
            "created_at": row["created_at"],
            "metadata": _json_loads(row["metadata_json"], {}),
        }

    def update_snapshot_metadata(
        self,
        snapshot_id: str,
        updates: Dict[str, Any],
        *,
        replace: bool = False,
    ) -> Dict[str, Any]:
        current = self.get_snapshot(snapshot_id)
        metadata = {} if replace else dict(current.get("metadata") or {})
        update_payload = dict(updates or {})
        if not replace and isinstance(metadata.get("extra"), dict) and isinstance(update_payload.get("extra"), dict):
            merged_extra = dict(metadata.get("extra") or {})
            merged_extra.update(dict(update_payload.get("extra") or {}))
            update_payload["extra"] = merged_extra
        metadata.update(update_payload)
        with self._connect() as conn:
            conn.execute(
                "UPDATE snapshots SET metadata_json = ? WHERE snapshot_id = ?",
                (_json_dumps(metadata), str(snapshot_id)),
            )
            conn.commit()
        return {
            "snapshot_id": current["snapshot_id"],
            "created_at": current["created_at"],
            "metadata": metadata,
        }

    def list_clusters(self, snapshot_id: Optional[str] = None, limit: int = 100, sort: str = "size_desc") -> List[Dict[str, Any]]:
        sid = self.resolve_snapshot_id(snapshot_id)
        order_by = {
            "size_desc": "size DESC, cluster_id ASC",
            "size_asc": "size ASC, cluster_id ASC",
            "cluster_id_asc": "cluster_id ASC",
            "cluster_id_desc": "cluster_id DESC",
        }.get(sort, "size DESC, cluster_id ASC")
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT cluster_id, size, metadata_json FROM clusters WHERE snapshot_id = ? ORDER BY {order_by} LIMIT ?",
                (sid, int(limit)),
            ).fetchall()
        return [
            {
                "cluster_id": int(r["cluster_id"]),
                "size": int(r["size"]),
                "metadata": _json_loads(r["metadata_json"], {}),
            }
            for r in rows
        ]

    def top_gaps(self, snapshot_id: Optional[str] = None, k: int = 25) -> List[Dict[str, Any]]:
        sid = self.resolve_snapshot_id(snapshot_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT gap_id, region_index, size, avg_gap_score, max_gap_score, density_z,
                       cluster_ids_json, metadata_json
                FROM gaps
                WHERE snapshot_id = ?
                ORDER BY COALESCE(density_z, avg_gap_score, max_gap_score, -1e9) DESC,
                         size DESC,
                         gap_id ASC
                LIMIT ?
                """,
                (sid, int(k)),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "gap_id": r["gap_id"],
                    "region_index": _to_int(r["region_index"]),
                    "size": int(r["size"]),
                    "avg_gap_score": _to_float(r["avg_gap_score"]),
                    "max_gap_score": _to_float(r["max_gap_score"]),
                    "density_z": _to_float(r["density_z"]),
                    "cluster_ids": _json_loads(r["cluster_ids_json"], []),
                    "metadata": _json_loads(r["metadata_json"], {}),
                }
            )
        return out

    def _paper_row_to_record(self, row: sqlite3.Row, include_embedding: bool = False) -> Dict[str, Any]:
        raw = _json_loads(row["raw_json"], {})
        if not isinstance(raw, dict):
            raw = {}
        rec = dict(raw)
        rec["paper_id"] = row["paper_id"]
        rec["title"] = row["title"]
        rec["abstract"] = row["abstract"]
        if row["publication_year"] is not None:
            rec["publication_year"] = int(row["publication_year"])
            rec["year"] = int(row["publication_year"])
        if row["doi"] is not None:
            rec["doi"] = row["doi"]
        if row["journal"] is not None:
            rec["journal"] = row["journal"]
        if row["cluster_id"] is not None:
            rec["cluster_id"] = int(row["cluster_id"])
        if row["gap_score"] is not None:
            rec["gap_score"] = float(row["gap_score"])
        if row["gap_region"] is not None:
            rec["gap_region"] = int(row["gap_region"])
        if include_embedding:
            emb = _to_embedding_list(row["embedding_json"]) if "embedding_json" in row.keys() else None
            if emb is not None:
                rec["embedding"] = emb
        return rec

    def _fetch_papers_by_ids(
        self,
        snapshot_id: str,
        paper_ids: Sequence[str],
        include_embedding: bool = False,
    ) -> List[Dict[str, Any]]:
        if not paper_ids:
            return []
        placeholders = ",".join("?" for _ in paper_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM papers WHERE snapshot_id = ? AND paper_id IN ({placeholders})",
                [snapshot_id, *paper_ids],
            ).fetchall()
        records = [self._paper_row_to_record(r, include_embedding=include_embedding) for r in rows]
        order = {pid: i for i, pid in enumerate(paper_ids)}
        records.sort(key=lambda r: order.get(str(r.get("paper_id")), 10**9))
        return records

    def _lookup_paper_id_alias_candidates(self, snapshot_id: str, paper_id: str) -> List[str]:
        text = str(paper_id or "").strip()
        if not text:
            return []

        prefix, raw_value = _split_paper_id_prefix(text)
        candidate_specs: List[Tuple[str, str]] = []
        if prefix and raw_value and "__src" not in raw_value:
            base = f"{prefix}:{raw_value}"
            candidate_specs.append((base, f"{_escape_sql_like(base)}__src%"))
        elif not prefix:
            for candidate_prefix in _PAPER_ID_ALIAS_PREFIXES:
                base = f"{candidate_prefix}:{text}"
                candidate_specs.append((base, f"{_escape_sql_like(base)}__src%"))

        clauses = ["paper_id = ?"]
        params: List[Any] = [snapshot_id, text]
        for exact_value, like_pattern in candidate_specs:
            clauses.append("paper_id = ?")
            params.append(exact_value)
            clauses.append("paper_id LIKE ? ESCAPE '\\'")
            params.append(like_pattern)

        sql = (
            "SELECT paper_id FROM papers WHERE snapshot_id = ? AND ("
            + " OR ".join(clauses)
            + ") ORDER BY row_index ASC"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        seen: set[str] = set()
        out: List[str] = []
        for row in rows:
            candidate = str(row["paper_id"] or "").strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
        return out

    def resolve_paper_ids(self, snapshot_id: Optional[str], paper_ids: Sequence[str]) -> Dict[str, Any]:
        sid = self.resolve_snapshot_id(snapshot_id)
        requested_paper_ids = _normalize_paper_ids(paper_ids)
        resolved_paper_ids: List[str] = []
        resolved_map: Dict[str, str] = {}
        unresolved_paper_ids: List[str] = []
        ambiguous_paper_ids: Dict[str, List[str]] = {}
        seen_resolved: set[str] = set()

        for requested_paper_id in requested_paper_ids:
            candidates = self._lookup_paper_id_alias_candidates(sid, requested_paper_id)
            if not candidates:
                unresolved_paper_ids.append(requested_paper_id)
                continue
            if len(candidates) > 1:
                ambiguous_paper_ids[requested_paper_id] = candidates
                continue
            canonical_paper_id = candidates[0]
            resolved_map[requested_paper_id] = canonical_paper_id
            if canonical_paper_id in seen_resolved:
                continue
            seen_resolved.add(canonical_paper_id)
            resolved_paper_ids.append(canonical_paper_id)

        return {
            "snapshot_id": sid,
            "requested_paper_ids": requested_paper_ids,
            "resolved_paper_ids": resolved_paper_ids,
            "resolved_map": resolved_map,
            "unresolved_paper_ids": unresolved_paper_ids,
            "ambiguous_paper_ids": ambiguous_paper_ids,
        }

    def papers_batch(
        self,
        snapshot_id: Optional[str],
        paper_ids: Sequence[str],
        fields: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        sid = self.resolve_snapshot_id(snapshot_id)
        include_embedding = bool(fields) and ("embedding" in set(fields))
        records = self._fetch_papers_by_ids(sid, [str(pid) for pid in paper_ids], include_embedding=include_embedding)
        if not fields:
            return records
        keep = set(fields)
        keep.add("paper_id")
        return [{k: rec.get(k) for k in keep} for rec in records]

    def _query_cluster_papers_sql(self, snapshot_id: str, cluster_id: int, limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM papers
                WHERE snapshot_id = ? AND cluster_id = ?
                ORDER BY COALESCE(gap_score, -1e9) DESC,
                         COALESCE(publication_year, -1) DESC,
                         row_index ASC
                LIMIT ?
                """,
                (snapshot_id, int(cluster_id), int(limit)),
            ).fetchall()
        return [self._paper_row_to_record(r) for r in rows]

    def _load_cluster_rows(self, snapshot_id: str, cluster_id: int) -> List[sqlite3.Row]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM papers
                WHERE snapshot_id = ? AND cluster_id = ?
                ORDER BY COALESCE(gap_score, -1e9) DESC,
                         COALESCE(publication_year, -1) DESC,
                         row_index ASC
                """,
                (snapshot_id, int(cluster_id)),
            ).fetchall()
        return list(rows)

    def _rows_with_embeddings(
        self,
        rows: Sequence[sqlite3.Row],
    ) -> Tuple[List[sqlite3.Row], Optional["np.ndarray"]]:  # type: ignore[name-defined]
        try:
            import numpy as np
        except Exception:
            return [], None

        parsed: List[Tuple[sqlite3.Row, List[float]]] = []
        dim_counts: Dict[int, int] = {}
        for row in rows:
            emb = _to_embedding_list(row["embedding_json"]) if "embedding_json" in row.keys() else None
            if emb is None:
                continue
            dim = len(emb)
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
            parsed.append((row, emb))

        if not parsed:
            return [], None

        # Use the most common dimension to handle mixed/partial snapshots safely.
        target_dim = max(dim_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        filtered_rows: List[sqlite3.Row] = []
        vectors: List[List[float]] = []
        for row, emb in parsed:
            if len(emb) != target_dim:
                continue
            filtered_rows.append(row)
            vectors.append(emb)
        if not vectors:
            return [], None
        return filtered_rows, np.asarray(vectors, dtype=float)

    def _cosine_distance_to_centroid(self, x: "np.ndarray", centroid: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
        import numpy as np

        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        c_norm = float(np.linalg.norm(centroid))
        if c_norm <= 1e-12:
            return np.linalg.norm(x - centroid, axis=1)
        x_safe = x / np.clip(x_norm, 1e-12, None)
        c_safe = centroid / c_norm
        return 1.0 - (x_safe @ c_safe)

    def _select_cluster_centroid_rows(self, rows: Sequence[sqlite3.Row], limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        rows_emb, x = self._rows_with_embeddings(rows)
        if x is None or len(rows_emb) == 0:
            return [self._paper_row_to_record(r) for r in list(rows)[:limit]]
        import numpy as np

        centroid = x.mean(axis=0)
        d = self._cosine_distance_to_centroid(x, centroid)
        order = np.argsort(d)
        selected = [rows_emb[int(i)] for i in order[: min(limit, len(order))]]

        if len(selected) < limit:
            selected_ids = {str(r["paper_id"]) for r in selected}
            for row in rows:
                pid = str(row["paper_id"])
                if pid in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(pid)
                if len(selected) >= limit:
                    break

        out = [self._paper_row_to_record(r) for r in selected[:limit]]
        for rec in out:
            rec.setdefault("selection_meta", {})
            if isinstance(rec["selection_meta"], dict):
                rec["selection_meta"].setdefault("sampling_mode", "centroid")
        return out

    def _cluster_centroid(self, rows: Sequence[sqlite3.Row]) -> Optional["np.ndarray"]:  # type: ignore[name-defined]
        rows_emb, x = self._rows_with_embeddings(rows)
        if x is None or len(rows_emb) == 0:
            return None
        return x.mean(axis=0)

    def _query_cluster_boundary_papers(
        self,
        snapshot_id: str,
        cluster_id: int,
        other_cluster_id: int,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        rows_self = self._load_cluster_rows(snapshot_id, cluster_id)
        rows_other = self._load_cluster_rows(snapshot_id, other_cluster_id)
        if not rows_self or not rows_other:
            return []

        rows_self_emb, x_self = self._rows_with_embeddings(rows_self)
        rows_other_emb, x_other = self._rows_with_embeddings(rows_other)
        if x_self is None or x_other is None or len(rows_self_emb) == 0 or len(rows_other_emb) == 0:
            return self._query_cluster_papers_sql(snapshot_id, cluster_id, limit)
        if x_self.shape[1] != x_other.shape[1]:
            return self._query_cluster_papers_sql(snapshot_id, cluster_id, limit)

        import numpy as np

        c_self = x_self.mean(axis=0)
        c_other = x_other.mean(axis=0)
        d_self = self._cosine_distance_to_centroid(x_self, c_self)
        d_other = self._cosine_distance_to_centroid(x_self, c_other)
        margin = np.abs(d_self - d_other)  # small margin => near boundary between cluster prototypes
        midpoint_closeness = 0.5 * (d_self + d_other)

        order = np.lexsort((d_self, midpoint_closeness, margin))
        selected_rows = [rows_self_emb[int(i)] for i in order[: min(limit, len(order))]]

        if len(selected_rows) < limit:
            selected_ids = {str(r["paper_id"]) for r in selected_rows}
            for row in rows_self:
                pid = str(row["paper_id"])
                if pid in selected_ids:
                    continue
                selected_rows.append(row)
                selected_ids.add(pid)
                if len(selected_rows) >= limit:
                    break

        out = [self._paper_row_to_record(r) for r in selected_rows[:limit]]
        for rec in out:
            rec.setdefault("selection_meta", {})
            if isinstance(rec["selection_meta"], dict):
                rec["selection_meta"].setdefault("sampling_mode", "cluster_pair_boundary")
                rec["selection_meta"].setdefault("other_cluster_id", int(other_cluster_id))
        return out

    def _query_cluster_papers(self, snapshot_id: str, cluster_id: int, limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        rows = self._load_cluster_rows(snapshot_id, cluster_id)
        if not rows:
            return []
        return self._select_cluster_centroid_rows(rows, limit)

    def _query_gap_papers(self, snapshot_id: str, gap_id: str, limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.*
                FROM gap_papers gp
                JOIN papers p
                  ON p.snapshot_id = gp.snapshot_id
                 AND p.paper_id = gp.paper_id
                WHERE gp.snapshot_id = ? AND gp.gap_id = ?
                ORDER BY COALESCE(gp.rank_idx, 1e9) ASC,
                         COALESCE(gp.gap_score, p.gap_score, -1e9) DESC
                LIMIT ?
                """,
                (snapshot_id, gap_id, int(limit)),
            ).fetchall()
        return [self._paper_row_to_record(r) for r in rows]

    def _query_diverse_papers(self, snapshot_id: str, exclude_ids: set[str], limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        params: List[Any] = [snapshot_id]
        not_in_sql = ""
        if exclude_ids:
            placeholders = ",".join("?" for _ in exclude_ids)
            not_in_sql = f"AND paper_id NOT IN ({placeholders})"
            params.extend(sorted(exclude_ids))
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM papers
                WHERE snapshot_id = ?
                  {not_in_sql}
                ORDER BY COALESCE(gap_score, -1e9) DESC,
                         COALESCE(publication_year, -1) DESC,
                         row_index ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._paper_row_to_record(r) for r in rows]

    def _query_counter_terms(self, snapshot_id: str, terms: Sequence[str], limit: int, exclude_ids: set[str]) -> List[Dict[str, Any]]:
        if limit <= 0 or not terms:
            return []
        params: List[Any] = [snapshot_id]
        term_clauses: List[str] = []
        for t in terms:
            t = str(t).strip()
            if not t:
                continue
            like = f"%{t.lower()}%"
            term_clauses.append("(LOWER(COALESCE(title,'')) LIKE ? OR LOWER(COALESCE(abstract,'')) LIKE ?)")
            params.extend([like, like])
        if not term_clauses:
            return []
        not_in_sql = ""
        if exclude_ids:
            placeholders = ",".join("?" for _ in exclude_ids)
            not_in_sql = f"AND paper_id NOT IN ({placeholders})"
            params.extend(sorted(exclude_ids))
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM papers
                WHERE snapshot_id = ?
                  AND ({' OR '.join(term_clauses)})
                  {not_in_sql}
                ORDER BY COALESCE(publication_year, -1) DESC,
                         COALESCE(gap_score, -1e9) DESC,
                         row_index ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._paper_row_to_record(r) for r in rows]

    def _get_gap_meta(self, snapshot_id: str, gap_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT gap_id, region_index, size, avg_gap_score, max_gap_score, density_z,
                       cluster_ids_json, metadata_json
                FROM gaps WHERE snapshot_id = ? AND gap_id = ?
                """,
                (snapshot_id, gap_id),
            ).fetchone()
        if not row:
            raise KeyError(f"Unknown gap_id '{gap_id}' for snapshot '{snapshot_id}'")
        return {
            "gap_id": row["gap_id"],
            "region_index": _to_int(row["region_index"]),
            "size": int(row["size"]),
            "avg_gap_score": _to_float(row["avg_gap_score"]),
            "max_gap_score": _to_float(row["max_gap_score"]),
            "density_z": _to_float(row["density_z"]),
            "cluster_ids": _json_loads(row["cluster_ids_json"], []),
            "metadata": _json_loads(row["metadata_json"], {}),
        }

    def _find_gap_ids_for_cluster_pair(self, snapshot_id: str, cluster_a: int, cluster_b: int) -> List[str]:
        ids: List[str] = []
        for g in self.top_gaps(snapshot_id=snapshot_id, k=10000):
            cluster_ids = {int(c) for c in (g.get("cluster_ids") or []) if _to_int(c) is not None}
            if int(cluster_a) in cluster_ids and int(cluster_b) in cluster_ids:
                ids.append(str(g["gap_id"]))
        return ids

    def _snapshot_embedding_scope(self, snapshot: Dict[str, Any]) -> str:
        metadata = dict((snapshot or {}).get("metadata") or {})
        candidates: List[str] = []
        if metadata.get("embedding_source") is not None:
            candidates.append(str(metadata.get("embedding_source") or ""))
        if metadata.get("embedding_name") is not None:
            candidates.append(str(metadata.get("embedding_name") or ""))
        analysis_config = metadata.get("analysis_config")
        if isinstance(analysis_config, dict):
            if analysis_config.get("embedding_name") is not None:
                candidates.append(str(analysis_config.get("embedding_name") or ""))
        for value in candidates:
            normalized = str(value).strip().lower()
            if normalized:
                return normalized
        return ""

    def _retrospective_cutoff_year(self, snapshot_id: str) -> Optional[int]:
        snapshot = self.get_snapshot(snapshot_id)
        metadata = dict(snapshot.get("metadata") or {})
        split_role = str(metadata.get("split_role") or "").strip().lower()
        if split_role != "historical":
            return None
        cutoff_date = str(metadata.get("cutoff_date") or "").strip()
        cutoff_year = _to_int(cutoff_date[:4]) if cutoff_date else None
        if cutoff_year is None:
            raise ValueError(
                f"Snapshot '{snapshot_id}' is historical but has no valid cutoff_date; "
                "cannot apply retrospective cue filtering safely."
            )
        return int(cutoff_year)

    def _cue_similarity_seed_int(self, seed_value: Any, request_payload: Dict[str, Any]) -> int:
        if isinstance(seed_value, int) and not isinstance(seed_value, bool):
            return int(seed_value)
        if seed_value is not None:
            seed_text = str(seed_value)
        else:
            stable_payload = dict(request_payload or {})
            stable_payload.pop("cue_similarity_seed", None)
            seed_text = _json_dumps(stable_payload)
        digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def _cue_query_text(self, discovery_cue: Any) -> str:
        cue = normalize_discovery_cue(discovery_cue)
        if cue is None:
            return ""
        parts: List[str] = []
        if cue.goal:
            parts.append(str(cue.goal))
        if cue.text:
            parts.append(str(cue.text))
        parts.extend([str(v) for v in (cue.include_terms or []) if str(v).strip()])
        for mapping in (cue.hard_constraints, cue.soft_constraints):
            if not isinstance(mapping, dict):
                continue
            for field, values in mapping.items():
                values_text = " ".join(str(v) for v in (values or []) if str(v).strip())
                if values_text:
                    parts.append(f"{field} {values_text}")
        fingerprint = cue.fingerprint if isinstance(cue.fingerprint, dict) else {}
        lexical_query = str(fingerprint.get("lexical_query") or "").strip()
        if lexical_query:
            parts.append(lexical_query)
        freeform_terms = [str(v) for v in (fingerprint.get("freeform_terms") or []) if str(v).strip()]
        if freeform_terms:
            parts.extend(freeform_terms[:12])
        query_text = " ".join(part for part in parts if str(part).strip()).strip()
        if query_text:
            return query_text[:4000]
        return str(cue.model_dump_json())[:4000]

    def _embed_texts_with_qwen(self, texts: Sequence[str]) -> List[List[float]]:
        try:
            from novelty_app.evaluation.qwen_client import QwenClient
        except Exception:
            from evaluation.qwen_client import QwenClient  # type: ignore

        base_url = str(os.getenv("QWEN_BASE_URL", "http://0.0.0.0:8000")).strip() or "http://0.0.0.0:8000"
        client = QwenClient(base_url=base_url)
        normalized_texts = [str(t or "").strip() for t in texts]
        return client.embed(normalized_texts, normalize=True)

    def _cue_similarity_source_matrix(self, snapshot_id: str) -> Dict[str, Any]:
        cached = self._cue_similarity_cache.get(snapshot_id)
        if cached is not None:
            return cached
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT paper_id, publication_year, embedding_json
                FROM papers
                WHERE snapshot_id = ?
                  AND embedding_json IS NOT NULL
                ORDER BY row_index ASC, paper_id ASC
                """,
                (snapshot_id,),
            ).fetchall()
        parsed: List[Tuple[str, Optional[int], List[float]]] = []
        dim_counts: Dict[int, int] = {}
        for row in rows:
            emb = _to_embedding_list(row["embedding_json"])
            if emb is None:
                continue
            dim = len(emb)
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
            parsed.append((str(row["paper_id"]), _to_int(row["publication_year"]), emb))
        if not parsed:
            raise ValueError(f"Cue source snapshot '{snapshot_id}' has no usable embeddings.")
        target_dim = max(dim_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        paper_ids: List[str] = []
        publication_years: List[Optional[int]] = []
        vectors: List[List[float]] = []
        for paper_id, publication_year, emb in parsed:
            if len(emb) != target_dim:
                continue
            paper_ids.append(paper_id)
            publication_years.append(publication_year)
            vectors.append(emb)
        if not vectors:
            raise ValueError(
                f"Cue source snapshot '{snapshot_id}' has no embeddings matching its dominant dimension."
            )
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("numpy is required for cue similarity retrieval.") from exc

        matrix = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-12, None)
        cached = {
            "snapshot_id": snapshot_id,
            "paper_ids": paper_ids,
            "publication_years": publication_years,
            "matrix": matrix,
        }
        self._cue_similarity_cache[snapshot_id] = cached
        return cached

    def _query_cue_full_similarity(
        self,
        *,
        active_snapshot_id: str,
        cue_source_snapshot_id: str,
        discovery_cue: Any,
        top_k: int,
        sample_n: int,
        seed_value: Any,
        request_payload: Dict[str, Any],
        selection_strategy: str = "sample",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if top_k < 1:
            raise ValueError("cue_similarity_top_k must be >= 1.")
        if sample_n < 0:
            raise ValueError("cue_similarity_sample_n must be >= 0.")
        if selection_strategy not in {"sample", "top"}:
            raise ValueError("cue similarity selection_strategy must be 'sample' or 'top'.")

        cue_source_snapshot = self.get_snapshot(cue_source_snapshot_id)
        source_scope = self._snapshot_embedding_scope(cue_source_snapshot)
        if source_scope != "qwen":
            raise ValueError(
                f"Cue source snapshot '{cue_source_snapshot_id}' uses embedding scope "
                f"'{source_scope or 'unknown'}'; v1 cue similarity requires qwen embeddings."
            )

        matrix_cache = self._cue_similarity_source_matrix(cue_source_snapshot_id)
        paper_ids = list(matrix_cache["paper_ids"])
        publication_years = list(matrix_cache["publication_years"])
        matrix = matrix_cache["matrix"]

        query_text = self._cue_query_text(discovery_cue)
        if not query_text:
            return [], {
                "source_snapshot_id": cue_source_snapshot_id,
                "candidate_count": 0,
                "top_k": int(top_k),
                "sample_n": int(sample_n),
                "filtered_by_cutoff": 0,
                "sampled_ids": [],
            }

        embeddings = self._embed_texts_with_qwen([query_text])
        if not embeddings:
            raise ValueError("Cue embedding endpoint returned no embeddings for discovery cue.")
        query_vector = _to_embedding_list(embeddings[0])
        if query_vector is None:
            raise ValueError("Cue embedding endpoint returned an invalid embedding payload.")

        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("numpy is required for cue similarity retrieval.") from exc

        query = np.asarray(query_vector, dtype=float)
        if matrix.shape[1] != query.shape[0]:
            raise ValueError(
                f"Cue embedding dim ({query.shape[0]}) does not match cue source snapshot "
                f"embedding dim ({matrix.shape[1]})."
            )
        query_norm = float(np.linalg.norm(query))
        if query_norm <= 1e-12:
            raise ValueError("Cue embedding vector norm is zero; cannot compute cosine similarity.")
        query = query / query_norm

        similarities = matrix @ query
        order = np.argsort(-similarities)
        top_count = min(int(top_k), len(order))
        top_candidates: List[Dict[str, Any]] = []
        for rank_idx, idx in enumerate(order[:top_count], start=1):
            pos = int(idx)
            top_candidates.append(
                {
                    "paper_id": str(paper_ids[pos]),
                    "publication_year": _to_int(publication_years[pos]),
                    "similarity": float(similarities[pos]),
                    "rank": rank_idx,
                }
            )

        cutoff_year = self._retrospective_cutoff_year(active_snapshot_id)
        filtered_by_cutoff = 0
        candidate_pool: List[Dict[str, Any]] = []
        for candidate in top_candidates:
            if cutoff_year is not None:
                year = _to_int(candidate.get("publication_year"))
                if year is None or int(year) > cutoff_year:
                    filtered_by_cutoff += 1
                    continue
            candidate_pool.append(candidate)

        sample_size = min(int(sample_n), len(candidate_pool))
        sampled_candidates: List[Dict[str, Any]] = []
        if sample_size > 0:
            if selection_strategy == "top":
                sampled_candidates = list(candidate_pool[:sample_size])
            else:
                seed_int = self._cue_similarity_seed_int(seed_value, request_payload)
                if sample_size >= len(candidate_pool):
                    sampled_candidates = list(candidate_pool)
                else:
                    rng = random.Random(seed_int)
                    sampled_candidates = rng.sample(candidate_pool, sample_size)

        sampled_ids = [str(candidate["paper_id"]) for candidate in sampled_candidates]
        sampled_records = self._fetch_papers_by_ids(cue_source_snapshot_id, sampled_ids)
        sampled_map = {str(candidate["paper_id"]): candidate for candidate in sampled_candidates}
        enriched_records: List[Dict[str, Any]] = []
        for record in sampled_records:
            paper_id = str(record.get("paper_id") or "")
            candidate = sampled_map.get(paper_id) or {}
            enriched = dict(record)
            selection_meta = dict(enriched.get("selection_meta") or {})
            selection_meta["cue_full_similarity_score"] = float(candidate.get("similarity") or 0.0)
            selection_meta["cue_full_similarity_rank"] = int(candidate.get("rank") or 0)
            selection_meta["cue_full_similarity_source_snapshot_id"] = cue_source_snapshot_id
            enriched["selection_meta"] = selection_meta
            enriched_records.append(enriched)

        return enriched_records, {
            "source_snapshot_id": cue_source_snapshot_id,
            "candidate_count": len(candidate_pool),
            "top_k": int(top_k),
            "sample_n": int(sample_n),
            "selection_strategy": selection_strategy,
            "filtered_by_cutoff": int(filtered_by_cutoff),
            "sampled_ids": sampled_ids,
        }

    def build_evidence_pack(self, req: Dict[str, Any]) -> Dict[str, Any]:
        snapshot_id = self.resolve_snapshot_id(req.get("snapshot_id"))
        target_type = str(req.get("target_type") or "")
        profile = str(req.get("profile") or "default").strip().lower() or "default"
        exemplars = max(0, int(req.get("exemplars", 25)))
        boundary = max(0, int(req.get("boundary", 25)))
        diverse = max(0, int(req.get("diverse", 25)))
        required_paper_ids = _normalize_paper_ids(req.get("required_paper_ids") or [])
        required_paper_source_snapshot_id = str(req.get("required_paper_source_snapshot_id") or "").strip() or None
        counter_queries = [str(q).strip() for q in (req.get("counter_queries") or []) if str(q).strip()]
        discovery_cue = normalize_discovery_cue(req.get("discovery_cue"))
        cue_queries = discovery_cue_query_terms(discovery_cue, max_queries=8) if discovery_cue is not None else []
        cue_source_snapshot_id = str(req.get("cue_source_snapshot_id") or "").strip() or None
        cue_similarity_top_k = int(req.get("cue_similarity_top_k", 50))
        cue_similarity_sample_n = int(req.get("cue_similarity_sample_n", 6))
        cue_similarity_seed = req.get("cue_similarity_seed")

        if profile == "focused_eval":
            exemplars = min(exemplars, 8) if exemplars else 8
            boundary = min(boundary, 8) if boundary else 8
            diverse = 0

        if target_type not in {"gap", "cluster_pair", "cue"}:
            raise ValueError("target_type must be 'gap', 'cluster_pair', or 'cue'")
        if discovery_cue is not None and not cue_source_snapshot_id:
            raise ValueError("cue_source_snapshot_id is required when discovery_cue is active.")
        if target_type == "cue" and discovery_cue is None:
            raise ValueError("discovery_cue is required for target_type='cue'.")
        if target_type == "cue":
            exemplars = 0
            boundary = 0
            diverse = 0

        resolved_required_paper_source_snapshot_id = snapshot_id
        if required_paper_source_snapshot_id:
            required_source_snapshot = self.get_snapshot(required_paper_source_snapshot_id)
            resolved_required_paper_source_snapshot_id = str(
                required_source_snapshot.get("snapshot_id") or required_paper_source_snapshot_id
            )

        selected: List[Dict[str, Any]] = []
        selected_by_id: Dict[str, Dict[str, Any]] = {}
        seen: set[str] = set()

        def add_many(records: Iterable[Dict[str, Any]], source: str, max_new: Optional[int] = None) -> int:
            added = 0
            for rec in records:
                if max_new is not None and added >= max_new:
                    break
                pid = str(rec.get("paper_id") or "")
                if not pid:
                    continue
                if pid in seen:
                    existing = selected_by_id.get(pid)
                    if existing is not None:
                        existing_sources = existing.get("selection_sources") or []
                        if not isinstance(existing_sources, list):
                            existing_sources = [str(existing_sources)]
                        if source not in existing_sources:
                            existing["selection_sources"] = [*existing_sources, source]
                        incoming_meta = rec.get("selection_meta")
                        if isinstance(incoming_meta, dict):
                            existing_meta = existing.get("selection_meta")
                            if not isinstance(existing_meta, dict):
                                existing_meta = {}
                            existing["selection_meta"] = {**existing_meta, **incoming_meta}
                    continue
                enriched = dict(rec)
                sources = enriched.get("selection_sources") or []
                if not isinstance(sources, list):
                    sources = [str(sources)]
                enriched["selection_sources"] = [*sources, source]
                selected.append(enriched)
                selected_by_id[pid] = enriched
                seen.add(pid)
                added += 1
            return added

        meta: Dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "target_type": target_type,
            "profile": profile,
            "sampling": {
                "cluster_exemplars": "centroid_if_embeddings_available_else_ranked_sql",
                "cluster_pair_boundary": "embedding_margin_if_embeddings_available_else_gap_and_ranked_sql",
            },
        }
        if discovery_cue is not None:
            meta["discovery_cue"] = discovery_cue.model_dump()
        if required_paper_ids:
            meta["required_paper_id_inputs"] = list(required_paper_ids)
            meta["required_paper_source_snapshot_id"] = resolved_required_paper_source_snapshot_id
            meta["required_paper_source_is_external"] = (
                str(resolved_required_paper_source_snapshot_id) != str(snapshot_id)
            )

        if required_paper_ids:
            resolution = self.resolve_paper_ids(resolved_required_paper_source_snapshot_id, required_paper_ids)
            unresolved_required_ids = list(resolution.get("unresolved_paper_ids") or [])
            ambiguous_required_ids = dict(resolution.get("ambiguous_paper_ids") or {})
            if unresolved_required_ids:
                raise ValueError(
                    "required_paper_ids not found in source snapshot "
                    f"`{resolved_required_paper_source_snapshot_id}`: {', '.join(unresolved_required_ids[:10])}"
                )
            if ambiguous_required_ids:
                alias, candidates = next(iter(ambiguous_required_ids.items()))
                raise ValueError(
                    "required_paper_ids are ambiguous in source snapshot "
                    f"`{resolved_required_paper_source_snapshot_id}` for `{alias}`: {', '.join(candidates[:10])}"
                )
            resolved_required_paper_ids = list(resolution.get("resolved_paper_ids") or [])
            meta["required_paper_ids"] = resolved_required_paper_ids
            meta["required_paper_id_aliases"] = dict(resolution.get("resolved_map") or {})
            required_records = self._fetch_papers_by_ids(
                resolved_required_paper_source_snapshot_id,
                resolved_required_paper_ids,
            )
            is_external_required_source = str(resolved_required_paper_source_snapshot_id) != str(snapshot_id)
            for record in required_records:
                selection_meta = dict(record.get("selection_meta") or {})
                selection_meta["required_paper_source_snapshot_id"] = resolved_required_paper_source_snapshot_id
                selection_meta["required_paper_source_is_external"] = bool(is_external_required_source)
                record["selection_meta"] = selection_meta
            add_many(required_records, "required_paper_id")

        if target_type == "gap":
            gap_id = str(req.get("gap_id") or "")
            if not gap_id:
                raise ValueError("gap_id is required for target_type='gap'")
            gap_meta = self._get_gap_meta(snapshot_id, gap_id)
            meta["gap"] = gap_meta
            add_many(self._query_gap_papers(snapshot_id, gap_id, boundary), "gap_boundary")

            touched_clusters = [int(c) for c in (gap_meta.get("cluster_ids") or []) if _to_int(c) is not None]
            if touched_clusters and exemplars > 0:
                per_cluster = max(1, exemplars // max(1, len(touched_clusters)))
                for cid in touched_clusters:
                    add_many(self._query_cluster_papers(snapshot_id, cid, per_cluster), f"cluster_{cid}_exemplar")

        elif target_type == "cluster_pair":
            cluster_a = _to_int(req.get("cluster_a"))
            cluster_b = _to_int(req.get("cluster_b"))
            if cluster_a is None or cluster_b is None:
                raise ValueError("cluster_a and cluster_b are required for target_type='cluster_pair'")
            meta["cluster_pair"] = {"cluster_a": cluster_a, "cluster_b": cluster_b}
            add_many(self._query_cluster_papers(snapshot_id, cluster_a, exemplars), f"cluster_{cluster_a}_exemplar")
            add_many(self._query_cluster_papers(snapshot_id, cluster_b, exemplars), f"cluster_{cluster_b}_exemplar")

            boundary_gap_ids = self._find_gap_ids_for_cluster_pair(snapshot_id, cluster_a, cluster_b)
            meta["boundary_gap_ids"] = boundary_gap_ids[:50]
            boundary_added = 0
            if boundary > 0:
                a_budget = (boundary + 1) // 2
                b_budget = boundary // 2
                boundary_added += add_many(
                    self._query_cluster_boundary_papers(snapshot_id, cluster_a, cluster_b, max(a_budget, a_budget * 3)),
                    f"cluster_{cluster_a}_boundary",
                    max_new=a_budget,
                )
                boundary_added += add_many(
                    self._query_cluster_boundary_papers(snapshot_id, cluster_b, cluster_a, max(b_budget, b_budget * 3)),
                    f"cluster_{cluster_b}_boundary",
                    max_new=b_budget,
                )

            remaining_boundary = max(0, boundary - boundary_added)
            if remaining_boundary > 0 and boundary_gap_ids:
                per_gap = max(1, remaining_boundary // max(1, len(boundary_gap_ids)))
                for gid in boundary_gap_ids:
                    if remaining_boundary <= 0:
                        break
                    added = add_many(self._query_gap_papers(snapshot_id, gid, per_gap), f"gap_{gid}_boundary", max_new=remaining_boundary)
                    remaining_boundary -= added

        else:
            # Cue-only retrieval intentionally omits graph targets. It supports a
            # conventional retrieve-then-generate baseline against the same
            # historical corpus and cue machinery used by the agentic methods.
            meta["cue_only_retrieval"] = True

        if diverse > 0:
            add_many(self._query_diverse_papers(snapshot_id, seen, diverse), "diverse")

        if discovery_cue is not None and cue_source_snapshot_id:
            cue_similarity_records, cue_similarity_stats = self._query_cue_full_similarity(
                active_snapshot_id=snapshot_id,
                cue_source_snapshot_id=cue_source_snapshot_id,
                discovery_cue=discovery_cue,
                top_k=cue_similarity_top_k,
                sample_n=cue_similarity_sample_n,
                selection_strategy="top" if target_type == "cue" else "sample",
                seed_value=cue_similarity_seed,
                request_payload=req,
            )
            added_after_dedupe = add_many(cue_similarity_records, "cue_full_similarity")
            cue_similarity_stats["added_after_dedupe"] = int(added_after_dedupe)
            meta["cue_full_similarity_stats"] = cue_similarity_stats

        if cue_queries:
            cue_query_limit = max(8, len(cue_queries) * 3)
            add_many(
                self._query_counter_terms(
                    snapshot_id,
                    cue_queries,
                    cue_query_limit + len(seen),
                    set(),
                ),
                "discovery_cue_query",
                max_new=cue_query_limit,
            )
            meta["discovery_cue_queries"] = cue_queries

        if counter_queries:
            add_many(
                self._query_counter_terms(snapshot_id, counter_queries, max(10, len(counter_queries) * 4), seen),
                "counter_query",
            )
            meta["counter_queries"] = counter_queries

        if discovery_cue is not None:
            cue_scores: List[float] = []
            cue_positive = 0
            for rec in selected:
                alignment = score_record_against_cue(rec, discovery_cue)
                score = float(alignment.get("score", 0.0) or 0.0)
                cue_scores.append(score)
                rec.setdefault("selection_meta", {})
                if isinstance(rec.get("selection_meta"), dict):
                    rec["selection_meta"]["cue_alignment"] = alignment
                    rec["selection_meta"]["cue_score"] = score
                if score > 0:
                    cue_positive += 1

            selected.sort(
                key=lambda rec: (
                    float(rec.get("selection_meta", {}).get("cue_score", 0.0) or 0.0),
                    float(rec.get("gap_score") or -1e9),
                    float(rec.get("publication_year") or -1),
                ),
                reverse=True,
            )
            cue_query_records: List[Dict[str, Any]] = []
            for rec in selected:
                sources = rec.get("selection_sources") or []
                if not isinstance(sources, list):
                    sources = [str(sources)]
                if "discovery_cue_query" in sources:
                    cue_query_records.append(rec)
            reserved_slots_applied = 0
            reserved_slots_requested = 0
            if cue_query_records:
                requested_total = max(1, exemplars + boundary + diverse)
                reserved_slots_requested = max(2, min(6, (requested_total + 3) // 4))
                reserved_slots_applied = min(len(cue_query_records), reserved_slots_requested)
                reserved_ids = {
                    str(rec.get("paper_id") or "") for rec in cue_query_records[:reserved_slots_applied] if rec.get("paper_id")
                }
                if reserved_ids:
                    # Keep cue-query papers near the front so freeform cue hits are not washed out by structural papers.
                    selected = [rec for rec in cue_query_records[:reserved_slots_applied]] + [
                        rec for rec in selected if str(rec.get("paper_id") or "") not in reserved_ids
                    ]
            meta["cue_stats"] = {
                "n_scored_papers": len(cue_scores),
                "n_positive_cue_matches": cue_positive,
                "max_cue_score": max(cue_scores) if cue_scores else 0.0,
                "avg_cue_score": (sum(cue_scores) / len(cue_scores)) if cue_scores else 0.0,
                "reserved_slots_requested": reserved_slots_requested,
                "reserved_slots_applied": reserved_slots_applied,
            }

        return {
            "snapshot_id": snapshot_id,
            "target_type": target_type,
            "papers": selected,
            "stats": {
                "n_papers": len(selected),
                "n_required_paper_ids": len(required_paper_ids),
                "required_paper_source_snapshot_id": (
                    resolved_required_paper_source_snapshot_id if required_paper_ids else None
                ),
                "n_counter_queries": len(counter_queries),
                "n_discovery_cue_queries": len(cue_queries),
                "requested": {"exemplars": exemplars, "boundary": boundary, "diverse": diverse},
            },
            "meta": meta,
            "discovery_cue": discovery_cue_to_dict(discovery_cue),
        }

    def store_artifact(
        self,
        kind: str,
        target: Dict[str, Any],
        payload: Dict[str, Any],
        snapshot_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        artifact_id = str(uuid.uuid4())
        sid = snapshot_id or (target or {}).get("snapshot_id")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts(artifact_id, snapshot_id, kind, created_at, target_json, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    sid,
                    kind,
                    _utc_now_iso(),
                    _json_dumps(target or {}),
                    _json_dumps(payload or {}),
                ),
            )
            conn.commit()
        return {"artifact_id": artifact_id, "snapshot_id": sid, "kind": kind}

    def get_artifact(self, artifact_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT artifact_id, snapshot_id, kind, created_at, target_json, payload_json
                FROM artifacts
                WHERE artifact_id = ?
                """,
                (str(artifact_id),),
            ).fetchone()
        if not row:
            raise ValueError(f"Artifact not found: {artifact_id}")
        return {
            "artifact_id": row["artifact_id"],
            "snapshot_id": row["snapshot_id"],
            "kind": row["kind"],
            "created_at": row["created_at"],
            "target": _json_loads(row["target_json"], {}),
            "payload": _json_loads(row["payload_json"], {}),
        }

    def list_artifacts(
        self,
        snapshot_id: Optional[str] = None,
        limit: int = 50,
        kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        sid = self.resolve_snapshot_id(snapshot_id) if snapshot_id else None
        with self._connect() as conn:
            if sid and kind:
                rows = conn.execute(
                    """
                    SELECT artifact_id, snapshot_id, kind, created_at, target_json, payload_json
                    FROM artifacts
                    WHERE snapshot_id = ? AND kind = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (sid, str(kind), int(limit)),
                ).fetchall()
            elif sid:
                rows = conn.execute(
                    """
                    SELECT artifact_id, snapshot_id, kind, created_at, target_json, payload_json
                    FROM artifacts
                    WHERE snapshot_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (sid, int(limit)),
                ).fetchall()
            elif kind:
                rows = conn.execute(
                    """
                    SELECT artifact_id, snapshot_id, kind, created_at, target_json, payload_json
                    FROM artifacts
                    WHERE kind = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (str(kind), int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT artifact_id, snapshot_id, kind, created_at, target_json, payload_json
                    FROM artifacts
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "artifact_id": r["artifact_id"],
                    "snapshot_id": r["snapshot_id"],
                    "kind": r["kind"],
                    "created_at": r["created_at"],
                    "target": _json_loads(r["target_json"], {}),
                    "payload": _json_loads(r["payload_json"], {}),
                }
            )
        return out

    def store_evaluation_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(payload.get("run_id") or uuid.uuid4())
        snapshot_id = payload.get("snapshot_id")
        created_at = str(payload.get("created_at") or _utc_now_iso())
        cutoff_date = _to_text(payload.get("cutoff_date"))
        future_window_start = _to_text(payload.get("future_window_start"))
        future_window_end = _to_text(payload.get("future_window_end"))
        method_names = payload.get("method_names") or []
        config = payload.get("config") or {}
        summary = payload.get("summary") or {}
        metrics = payload.get("metrics") or {}
        observability = payload.get("observability") or {}
        status = str(payload.get("status") or "completed")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs(
                    run_id, snapshot_id, created_at, cutoff_date, future_window_start, future_window_end,
                    method_names_json, config_json, summary_json, metrics_json, observability_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _to_text(snapshot_id),
                    created_at,
                    cutoff_date,
                    future_window_start,
                    future_window_end,
                    _json_dumps(method_names),
                    _json_dumps(config),
                    _json_dumps(summary),
                    _json_dumps(metrics),
                    _json_dumps(observability),
                    status,
                ),
            )
            conn.commit()
        return {"run_id": run_id, "snapshot_id": snapshot_id, "status": status}

    def list_evaluation_runs(
        self,
        snapshot_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        sid = self.resolve_snapshot_id(snapshot_id) if snapshot_id else None
        with self._connect() as conn:
            if sid:
                rows = conn.execute(
                    """
                    SELECT run_id, snapshot_id, created_at, cutoff_date, future_window_start, future_window_end,
                           method_names_json, config_json, summary_json, metrics_json, observability_json, status
                    FROM evaluation_runs
                    WHERE snapshot_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (sid, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT run_id, snapshot_id, created_at, cutoff_date, future_window_start, future_window_end,
                           method_names_json, config_json, summary_json, metrics_json, observability_json, status
                    FROM evaluation_runs
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            config = _json_loads(r["config_json"], {})
            out.append(
                {
                    "run_id": r["run_id"],
                    "snapshot_id": r["snapshot_id"],
                    "created_at": r["created_at"],
                    "cutoff_date": r["cutoff_date"],
                    "future_window_start": r["future_window_start"],
                    "future_window_end": r["future_window_end"],
                    "method_names": _json_loads(r["method_names_json"], []),
                    "config": config,
                    "summary": _json_loads(r["summary_json"], {}),
                    "metrics": _json_loads(r["metrics_json"], {}),
                    "status": r["status"],
                    "discovery_cue": config.get("discovery_cue", {}),
                    "observability": _json_loads(r["observability_json"], {}),
                }
            )
        return out

    def store_evaluation_matches_batch(self, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        inserted = 0
        with self._connect() as conn:
            for rec in records:
                match_id = str(rec.get("match_id") or uuid.uuid4())
                conn.execute(
                    """
                    INSERT OR REPLACE INTO evaluation_matches(
                        match_id, run_id, snapshot_id, created_at, target_id, target_type, method_name, seed,
                        hypothesis_id, classification, historical_label, future_label, first_future_year,
                        historical_best_paper_id, future_best_paper_id, support_citations_json, hypothesis_json,
                        idea_scores_json, fingerprint_json, historical_match_json, future_match_json,
                        recovery_label, future_neighbor_label, gold_future_paper_id, gold_future_title, gold_future_year,
                        assigned_target_id, assigned_target_score, gold_rank, gold_reciprocal_rank, gold_hit_at_1,
                        gold_hit_at_5, gold_hit_at_10, cue_score, cue_weighted_rr, best_future_neighbor_paper_id,
                        best_historical_confounder_id, evidence_pack_summary_json, historical_candidates_json,
                        future_candidates_json, trace_ref_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        match_id,
                        str(rec.get("run_id") or ""),
                        _to_text(rec.get("snapshot_id")),
                        str(rec.get("created_at") or _utc_now_iso()),
                        str(rec.get("target_id") or ""),
                        str(rec.get("target_type") or ""),
                        str(rec.get("method_name") or ""),
                        _to_int(rec.get("seed")),
                        str(rec.get("hypothesis_id") or ""),
                        str(rec.get("recovery_label") or rec.get("classification") or ""),
                        _to_text(rec.get("historical_label")),
                        _to_text(rec.get("future_neighbor_label") or rec.get("future_label")),
                        _to_int(rec.get("gold_future_year") or rec.get("first_future_year")),
                        _to_text(rec.get("best_historical_confounder_id") or rec.get("historical_best_paper_id")),
                        _to_text(rec.get("best_future_neighbor_paper_id") or rec.get("future_best_paper_id")),
                        _json_dumps(rec.get("support_citations") or []),
                        _json_dumps(rec.get("hypothesis") or {}),
                        _json_dumps(rec.get("idea_scores") or {}),
                        _json_dumps(rec.get("fingerprint") or {}),
                        _json_dumps(rec.get("historical_match") or {}),
                        _json_dumps(rec.get("future_match") or {}),
                        str(rec.get("recovery_label") or rec.get("classification") or ""),
                        _to_text(rec.get("future_neighbor_label") or rec.get("future_label")),
                        str(rec.get("gold_future_paper_id") or rec.get("future_best_paper_id") or ""),
                        _to_text(rec.get("gold_future_title")),
                        _to_int(rec.get("gold_future_year") or rec.get("first_future_year")),
                        str(rec.get("assigned_target_id") or rec.get("target_id") or ""),
                        _to_float(rec.get("assigned_target_score")),
                        _to_int(rec.get("gold_rank")),
                        _to_float(rec.get("gold_reciprocal_rank")),
                        int(bool(rec.get("gold_hit_at_1"))),
                        int(bool(rec.get("gold_hit_at_5"))),
                        int(bool(rec.get("gold_hit_at_10"))),
                        _to_float(rec.get("cue_score")),
                        _to_float(rec.get("cue_weighted_rr")),
                        _to_text(rec.get("best_future_neighbor_paper_id") or rec.get("future_best_paper_id")),
                        _to_text(rec.get("best_historical_confounder_id") or rec.get("historical_best_paper_id")),
                        _json_dumps(rec.get("evidence_pack_summary") or {}),
                        _json_dumps(rec.get("historical_candidates") or []),
                        _json_dumps(rec.get("future_candidates") or []),
                        _json_dumps(rec.get("trace_ref") or {}),
                    ),
                )
                inserted += 1
            conn.commit()
        return {"stored": inserted}

    def list_evaluation_matches(
        self,
        *,
        run_id: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        limit: int = 250,
    ) -> List[Dict[str, Any]]:
        sid = self.resolve_snapshot_id(snapshot_id) if snapshot_id else None
        params: List[Any] = []
        where: List[str] = []
        if run_id:
            where.append("run_id = ?")
            params.append(str(run_id))
        if sid:
            where.append("snapshot_id = ?")
            params.append(sid)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT match_id, run_id, snapshot_id, created_at, target_id, target_type, method_name, seed,
                       hypothesis_id, classification, historical_label, future_label, first_future_year,
                       historical_best_paper_id, future_best_paper_id, support_citations_json, hypothesis_json,
                       idea_scores_json, fingerprint_json, historical_match_json, future_match_json,
                       recovery_label, future_neighbor_label, gold_future_paper_id, gold_future_title, gold_future_year,
                       assigned_target_id, assigned_target_score, gold_rank, gold_reciprocal_rank, gold_hit_at_1,
                       gold_hit_at_5, gold_hit_at_10, cue_score, cue_weighted_rr, best_future_neighbor_paper_id,
                       best_historical_confounder_id, evidence_pack_summary_json, historical_candidates_json,
                       future_candidates_json, trace_ref_json
                FROM evaluation_matches
                {where_sql}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            hypothesis = _json_loads(r["hypothesis_json"], {})
            historical_match = _json_loads(r["historical_match_json"], {})
            future_match = _json_loads(r["future_match_json"], {})
            historical_candidates = _json_loads(r["historical_candidates_json"], [])
            future_candidates = _json_loads(r["future_candidates_json"], [])
            out.append(
                {
                    "match_id": r["match_id"],
                    "run_id": r["run_id"],
                    "snapshot_id": r["snapshot_id"],
                    "created_at": r["created_at"],
                    "target_id": r["target_id"],
                    "target_type": r["target_type"],
                    "method_name": r["method_name"],
                    "seed": _to_int(r["seed"]) or 0,
                    "hypothesis_id": r["hypothesis_id"],
                    "recovery_label": r["recovery_label"] or r["classification"] or "not_recovered",
                    "historical_label": r["historical_label"] or "no_match",
                    "future_neighbor_label": r["future_neighbor_label"] or r["future_label"] or "no_match",
                    "gold_future_paper_id": r["gold_future_paper_id"] or r["future_best_paper_id"] or "",
                    "gold_future_title": r["gold_future_title"] or "",
                    "gold_future_year": _to_int(r["gold_future_year"]) or _to_int(r["first_future_year"]),
                    "assigned_target_id": r["assigned_target_id"] or r["target_id"],
                    "assigned_target_score": _to_float(r["assigned_target_score"]) or 0.0,
                    "gold_rank": _to_int(r["gold_rank"]),
                    "gold_reciprocal_rank": _to_float(r["gold_reciprocal_rank"]) or 0.0,
                    "gold_hit_at_1": bool(_to_int(r["gold_hit_at_1"])),
                    "gold_hit_at_5": bool(_to_int(r["gold_hit_at_5"])),
                    "gold_hit_at_10": bool(_to_int(r["gold_hit_at_10"])),
                    "cue_score": _to_float(r["cue_score"]),
                    "cue_weighted_rr": _to_float(r["cue_weighted_rr"]) or 0.0,
                    "best_future_neighbor_paper_id": r["best_future_neighbor_paper_id"] or r["future_best_paper_id"],
                    "best_historical_confounder_id": r["best_historical_confounder_id"] or r["historical_best_paper_id"],
                    "support_citations": _json_loads(r["support_citations_json"], []),
                    "hypothesis": hypothesis,
                    "idea_scores": _json_loads(r["idea_scores_json"], (hypothesis or {}).get("idea_scores", {})),
                    "fingerprint": _json_loads(r["fingerprint_json"], {}),
                    "evidence_pack_summary": _json_loads(r["evidence_pack_summary_json"], {}),
                    "historical_match": historical_match,
                    "future_match": future_match,
                    "historical_candidates": historical_candidates or ([historical_match] if historical_match else []),
                    "future_candidates": future_candidates or ([future_match] if future_match else []),
                    "discovery_cue": (hypothesis or {}).get("discovery_cue", {}),
                    "trace_ref": _json_loads(r["trace_ref_json"], (hypothesis or {}).get("trace_ref", {})),
                }
            )
        return out


def default_store() -> KnowledgeStore:
    return KnowledgeStore(default_db_path())

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from novelty_app.agents.knowledge_store import KnowledgeStore


class KnowledgeStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test.sqlite"
        self.store = KnowledgeStore(self.db_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_publish_snapshot_preserves_float_scores(self) -> None:
        payload = {
            "snapshot_id": "snap_1",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test"},
            "papers": [
                {
                    "paper_id": "p1",
                    "title": "Paper 1",
                    "abstract": "Abstract",
                    "publication_year": 2020,
                    "cluster_id": 1,
                    "gap_score": "0.75",
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
            "clusters": [{"cluster_id": 1, "size": 1, "metadata": {}}],
            "gaps": [{"gap_id": "gap_0", "region_index": 0, "size": 1, "avg_gap_score": "0.75", "max_gap_score": "0.75", "cluster_ids": [1], "metadata": {}}],
            "gap_papers": [{"gap_id": "gap_0", "paper_id": "p1", "rank": 0, "gap_score": "0.75"}],
            "llm_analyses": [],
        }
        self.store.publish_snapshot(payload)
        top_gap = self.store.top_gaps(snapshot_id="snap_1", k=1)[0]
        self.assertEqual(top_gap["avg_gap_score"], 0.75)
        paper = self.store.papers_batch(snapshot_id="snap_1", paper_ids=["p1"])[0]
        self.assertEqual(paper["gap_score"], 0.75)

    def test_resolve_paper_ids_accepts_unique_aliases(self) -> None:
        payload = {
            "snapshot_id": "snap_aliases",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test"},
            "papers": [
                {
                    "paper_id": "id:8719955__src2",
                    "title": "Paper 1",
                    "abstract": "Abstract 1",
                    "publication_year": 2020,
                    "cluster_id": 1,
                },
                {
                    "paper_id": "pmid:18445731",
                    "title": "Paper 2",
                    "abstract": "Abstract 2",
                    "publication_year": 2021,
                    "cluster_id": 1,
                },
            ],
            "clusters": [{"cluster_id": 1, "size": 2, "metadata": {}}],
            "gaps": [],
            "gap_papers": [],
            "llm_analyses": [],
        }

        self.store.publish_snapshot(payload)

        resolution = self.store.resolve_paper_ids(
            "snap_aliases",
            ["8719955", "pmid:18445731", "id:8719955", "id:8719955__src2"],
        )

        self.assertEqual(
            resolution["resolved_paper_ids"],
            ["id:8719955__src2", "pmid:18445731"],
        )
        self.assertEqual(resolution["resolved_map"]["8719955"], "id:8719955__src2")
        self.assertEqual(resolution["resolved_map"]["id:8719955"], "id:8719955__src2")
        self.assertEqual(resolution["resolved_map"]["id:8719955__src2"], "id:8719955__src2")
        self.assertEqual(resolution["unresolved_paper_ids"], [])
        self.assertEqual(resolution["ambiguous_paper_ids"], {})

    def test_resolve_paper_ids_reports_ambiguous_bare_alias(self) -> None:
        payload = {
            "snapshot_id": "snap_alias_ambiguous",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test"},
            "papers": [
                {
                    "paper_id": "id:9165532__src2",
                    "title": "Paper 1",
                    "abstract": "Abstract 1",
                    "publication_year": 2020,
                    "cluster_id": 1,
                },
                {
                    "paper_id": "pmid:9165532__src7",
                    "title": "Paper 2",
                    "abstract": "Abstract 2",
                    "publication_year": 2021,
                    "cluster_id": 1,
                },
            ],
            "clusters": [{"cluster_id": 1, "size": 2, "metadata": {}}],
            "gaps": [],
            "gap_papers": [],
            "llm_analyses": [],
        }

        self.store.publish_snapshot(payload)

        resolution = self.store.resolve_paper_ids("snap_alias_ambiguous", ["9165532"])

        self.assertEqual(resolution["resolved_paper_ids"], [])
        self.assertEqual(resolution["unresolved_paper_ids"], [])
        self.assertEqual(
            resolution["ambiguous_paper_ids"]["9165532"],
            ["id:9165532__src2", "pmid:9165532__src7"],
        )

    def test_build_evidence_pack_resolves_required_paper_id_aliases(self) -> None:
        payload = {
            "snapshot_id": "snap_required_alias",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test"},
            "papers": [
                {
                    "paper_id": "id:8719955__src2",
                    "title": "Paper 1",
                    "abstract": "Abstract 1",
                    "publication_year": 2020,
                    "cluster_id": 1,
                    "gap_score": 0.9,
                },
                {
                    "paper_id": "id:9999999__src4",
                    "title": "Paper 2",
                    "abstract": "Abstract 2",
                    "publication_year": 2021,
                    "cluster_id": 1,
                    "gap_score": 0.8,
                },
            ],
            "clusters": [{"cluster_id": 1, "size": 2, "metadata": {}}],
            "gaps": [{"gap_id": "gap_0", "region_index": 0, "size": 2, "avg_gap_score": 0.85, "max_gap_score": 0.9, "cluster_ids": [1], "metadata": {}}],
            "gap_papers": [
                {"gap_id": "gap_0", "paper_id": "id:8719955__src2", "rank": 0, "gap_score": 0.9},
                {"gap_id": "gap_0", "paper_id": "id:9999999__src4", "rank": 1, "gap_score": 0.8},
            ],
            "llm_analyses": [],
        }

        self.store.publish_snapshot(payload)

        evidence = self.store.build_evidence_pack(
            {
                "snapshot_id": "snap_required_alias",
                "target_type": "gap",
                "gap_id": "gap_0",
                "required_paper_ids": ["8719955"],
                "required_paper_source_snapshot_id": "snap_required_alias",
                "profile": "focused_eval",
                "exemplars": 1,
                "boundary": 1,
                "diverse": 0,
            }
        )

        self.assertEqual(evidence["meta"]["required_paper_ids"], ["id:8719955__src2"])
        self.assertEqual(evidence["meta"]["required_paper_id_inputs"], ["8719955"])
        self.assertEqual(evidence["meta"]["required_paper_id_aliases"]["8719955"], "id:8719955__src2")
        self.assertEqual(evidence["papers"][0]["paper_id"], "id:8719955__src2")

    def test_publish_snapshot_recreates_schema_after_db_truncation(self) -> None:
        self.db_path.write_bytes(b"")
        payload = {
            "snapshot_id": "snap_recovered",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test"},
            "papers": [
                {
                    "paper_id": "p1",
                    "title": "Recovered paper",
                    "abstract": "Recovered abstract",
                    "publication_year": 2020,
                    "cluster_id": 1,
                }
            ],
            "clusters": [{"cluster_id": 1, "size": 1, "metadata": {}}],
            "gaps": [],
            "gap_papers": [],
            "llm_analyses": [],
        }

        self.store.publish_snapshot(payload)

        paper = self.store.papers_batch(snapshot_id="snap_recovered", paper_ids=["p1"])[0]
        self.assertEqual(paper["paper_id"], "p1")
        with sqlite3.connect(self.db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
                ).fetchall()
            }
        self.assertIn("papers", tables)
        self.assertIn("snapshots", tables)

    def test_store_and_list_evaluation_records(self) -> None:
        run = {
            "run_id": "run_1",
            "snapshot_id": "snap_1",
            "created_at": "2026-03-11T00:00:00+00:00",
            "cutoff_date": "2020-12-31",
            "future_window_start": "2022-01-01",
            "future_window_end": "2025-12-31",
            "method_names": ["heuristic"],
            "config": {"x": 1},
            "summary": {"n": 1},
            "metrics": {"gold_recall_at_1": 1.0},
            "status": "completed",
            "observability": {
                "provider": "langfuse",
                "trace_id": "trace_run_1",
                "url": "https://langfuse.example/project/p/traces/trace_run_1",
            },
        }
        self.store.store_evaluation_run(run)
        self.store.store_evaluation_matches_batch(
            [
                {
                    "run_id": "run_1",
                    "snapshot_id": "snap_1",
                    "target_id": "cluster_pair_1_2",
                    "target_type": "cluster_pair",
                    "method_name": "heuristic",
                    "seed": 0,
                    "hypothesis_id": "h1",
                    "recovery_label": "gold_recovered",
                    "historical_label": "no_match",
                    "future_neighbor_label": "strong_match",
                    "gold_future_paper_id": "p9",
                    "gold_future_title": "Future Paper",
                    "gold_future_year": 2023,
                    "assigned_target_id": "cluster_pair_1_2",
                    "assigned_target_score": 0.91,
                    "gold_rank": 1,
                    "gold_reciprocal_rank": 1.0,
                    "gold_hit_at_1": True,
                    "gold_hit_at_5": True,
                    "gold_hit_at_10": True,
                    "best_historical_confounder_id": None,
                    "best_future_neighbor_paper_id": "p9",
                    "support_citations": ["p1"],
                    "hypothesis": {"title": "Hypothesis"},
                    "idea_scores": {"importance": {"score": 4}, "average_score": 4.0},
                    "fingerprint": {"material": ["liposome"]},
                    "evidence_pack_summary": {"n_papers": 2},
                    "historical_match": {},
                    "future_match": {"paper_id": "p9"},
                    "historical_candidates": [],
                    "future_candidates": [{"paper_id": "p9"}],
                    "trace_ref": {
                        "provider": "langfuse",
                        "trace_id": "trace_match_1",
                        "url": "https://langfuse.example/project/p/traces/trace_match_1",
                    },
                }
            ]
        )

        runs = self.store.list_evaluation_runs(limit=10)
        matches = self.store.list_evaluation_matches(run_id="run_1", limit=10)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], "run_1")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["recovery_label"], "gold_recovered")
        self.assertEqual(matches[0]["idea_scores"]["importance"]["score"], 4)
        self.assertEqual(matches[0]["gold_future_paper_id"], "p9")
        self.assertEqual(matches[0]["assigned_target_id"], "cluster_pair_1_2")
        self.assertEqual(runs[0]["observability"]["trace_id"], "trace_run_1")
        self.assertEqual(matches[0]["trace_ref"]["trace_id"], "trace_match_1")

    def test_update_snapshot_metadata_merges_fields(self) -> None:
        payload = {
            "snapshot_id": "snap_meta",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "streamlit_agent_console", "selected_clustering": "kmeans"},
            "papers": [],
            "clusters": [],
            "gaps": [],
            "gap_papers": [],
            "llm_analyses": [],
        }
        self.store.publish_snapshot(payload)

        updated = self.store.update_snapshot_metadata(
            "snap_meta",
            {
                "split_role": "historical",
                "cutoff_date": "2020-12-31",
                "future_window_start": "2022-01-01",
            },
        )

        self.assertEqual(updated["metadata"]["source"], "streamlit_agent_console")
        self.assertEqual(updated["metadata"]["split_role"], "historical")
        self.assertEqual(updated["metadata"]["cutoff_date"], "2020-12-31")
        self.assertEqual(updated["metadata"]["future_window_start"], "2022-01-01")

    def test_build_evidence_pack_applies_discovery_cue(self) -> None:
        payload = {
            "snapshot_id": "snap_cue",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test", "embedding_source": "qwen"},
            "papers": [
                {
                    "paper_id": "p1",
                    "title": "Folate liposome siRNA for breast cancer",
                    "abstract": "folate liposome sirna gene silencing in breast cancer",
                    "publication_year": 2020,
                    "cluster_id": 1,
                    "gap_score": 0.9,
                    "embedding": [1.0, 0.0, 0.0],
                },
                {
                    "paper_id": "p2",
                    "title": "Gold imaging in melanoma",
                    "abstract": "gold nanoparticle imaging in melanoma",
                    "publication_year": 2020,
                    "cluster_id": 1,
                    "gap_score": 0.6,
                    "embedding": [0.0, 1.0, 0.0],
                },
            ],
            "clusters": [{"cluster_id": 1, "size": 2, "metadata": {}}],
            "gaps": [{"gap_id": "gap_0", "region_index": 0, "size": 2, "avg_gap_score": 0.8, "max_gap_score": 0.9, "cluster_ids": [1], "metadata": {}}],
            "gap_papers": [
                {"gap_id": "gap_0", "paper_id": "p1", "rank": 0, "gap_score": 0.9},
                {"gap_id": "gap_0", "paper_id": "p2", "rank": 1, "gap_score": 0.6},
            ],
            "llm_analyses": [],
        }
        self.store.publish_snapshot(payload)
        self.store._embed_texts_with_qwen = lambda texts: [[1.0, 0.0, 0.0]]  # type: ignore[method-assign]

        evidence = self.store.build_evidence_pack(
            {
                "snapshot_id": "snap_cue",
                "target_type": "gap",
                "gap_id": "gap_0",
                "profile": "focused_eval",
                "exemplars": 2,
                "boundary": 2,
                "diverse": 10,
                "discovery_cue": {
                    "text": "Focus on folate liposome siRNA approaches in breast cancer",
                    "soft_constraints": {
                        "material": ["liposome"],
                        "payload": ["sirna"],
                        "targeting": ["folate"],
                        "disease": ["breast"],
                    },
                },
                "cue_source_snapshot_id": "snap_cue",
                "cue_similarity_top_k": 2,
                "cue_similarity_sample_n": 2,
                "cue_similarity_seed": "test_seed",
            }
        )

        self.assertEqual(evidence["papers"][0]["paper_id"], "p1")
        self.assertEqual(evidence["meta"]["discovery_cue"]["text"], "Focus on folate liposome siRNA approaches in breast cancer")
        self.assertEqual(evidence["meta"]["profile"], "focused_eval")
        self.assertEqual(evidence["stats"]["requested"]["diverse"], 0)
        self.assertGreater(
            float(evidence["papers"][0]["selection_meta"]["cue_alignment"]["score"]),
            float(evidence["papers"][1]["selection_meta"]["cue_alignment"]["score"]),
        )

    def test_build_evidence_pack_compiles_freeform_cue_and_reserves_cue_slots(self) -> None:
        payload = {
            "snapshot_id": "snap_freeform_cue",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test", "embedding_source": "qwen"},
            "papers": [
                {
                    "paper_id": "p1",
                    "title": "Folate liposome siRNA for breast cancer",
                    "abstract": "folate liposome sirna gene silencing in breast cancer",
                    "publication_year": 2020,
                    "cluster_id": 1,
                    "gap_score": 0.9,
                    "embedding": [1.0, 0.0, 0.0],
                },
                {
                    "paper_id": "p2",
                    "title": "Gold imaging in melanoma",
                    "abstract": "gold nanoparticle imaging in melanoma",
                    "publication_year": 2020,
                    "cluster_id": 1,
                    "gap_score": 0.6,
                    "embedding": [0.0, 1.0, 0.0],
                },
                {
                    "paper_id": "p3",
                    "title": "Surface coating design for inorganic nanoparticles in biofilms",
                    "abstract": "A polymer coating improves inorganic nanoparticle penetration into bacterial biofilms.",
                    "publication_year": 2021,
                    "cluster_id": 2,
                    "gap_score": 0.4,
                    "embedding": [0.0, 0.5, 0.5],
                },
            ],
            "clusters": [
                {"cluster_id": 1, "size": 2, "metadata": {}},
                {"cluster_id": 2, "size": 1, "metadata": {}},
            ],
            "gaps": [{"gap_id": "gap_0", "region_index": 0, "size": 2, "avg_gap_score": 0.8, "max_gap_score": 0.9, "cluster_ids": [1], "metadata": {}}],
            "gap_papers": [
                {"gap_id": "gap_0", "paper_id": "p1", "rank": 0, "gap_score": 0.9},
                {"gap_id": "gap_0", "paper_id": "p2", "rank": 1, "gap_score": 0.6},
            ],
            "llm_analyses": [],
        }
        self.store.publish_snapshot(payload)
        self.store._embed_texts_with_qwen = lambda texts: [[0.0, 0.5, 0.5]]  # type: ignore[method-assign]

        evidence = self.store.build_evidence_pack(
            {
                "snapshot_id": "snap_freeform_cue",
                "target_type": "gap",
                "gap_id": "gap_0",
                "profile": "focused_eval",
                "exemplars": 2,
                "boundary": 1,
                "diverse": 0,
                "discovery_cue": {
                    "text": "What characteristics should a coating for inorganic nanoparticles have to overcome biofilms?"
                },
                "cue_source_snapshot_id": "snap_freeform_cue",
                "cue_similarity_top_k": 3,
                "cue_similarity_sample_n": 2,
                "cue_similarity_seed": "biofilm_seed",
            }
        )

        self.assertTrue(evidence["meta"]["discovery_cue_queries"])
        self.assertIn("biofilm", evidence["meta"]["discovery_cue_queries"])
        self.assertEqual(evidence["papers"][0]["paper_id"], "p3")
        self.assertGreater(float(evidence["papers"][0]["selection_meta"]["cue_score"]), 0.0)
        self.assertGreater(evidence["meta"]["cue_stats"]["reserved_slots_applied"], 0)

        cue_only = self.store.build_evidence_pack(
            {
                "snapshot_id": "snap_freeform_cue",
                "target_type": "cue",
                "profile": "focused_eval",
                "discovery_cue": {
                    "text": "What characteristics should a coating for inorganic nanoparticles have to overcome biofilms?"
                },
                "cue_source_snapshot_id": "snap_freeform_cue",
                "cue_similarity_top_k": 3,
                "cue_similarity_sample_n": 2,
                "cue_similarity_seed": "biofilm_seed",
            }
        )

        self.assertTrue(cue_only["meta"]["cue_only_retrieval"])
        self.assertEqual(cue_only["meta"]["cue_full_similarity_stats"]["selection_strategy"], "top")
        self.assertEqual(cue_only["meta"]["cue_full_similarity_stats"]["sampled_ids"], ["p3", "p2"])
        self.assertEqual(cue_only["stats"]["requested"], {"exemplars": 0, "boundary": 0, "diverse": 0})
        self.assertTrue(cue_only["papers"])
        for paper in cue_only["papers"]:
            sources = list(paper.get("selection_sources") or [])
            self.assertFalse(any("cluster_" in source or "gap_" in source for source in sources))

    def test_build_evidence_pack_requires_cue_source_snapshot_when_cue_active(self) -> None:
        payload = {
            "snapshot_id": "snap_cue_required",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test", "embedding_source": "qwen"},
            "papers": [
                {
                    "paper_id": "p1",
                    "title": "Paper 1",
                    "abstract": "Cue paper",
                    "publication_year": 2020,
                    "cluster_id": 1,
                    "embedding": [1.0, 0.0],
                }
            ],
            "clusters": [{"cluster_id": 1, "size": 1, "metadata": {}}],
            "gaps": [{"gap_id": "gap_0", "region_index": 0, "size": 1, "avg_gap_score": 0.5, "max_gap_score": 0.5, "cluster_ids": [1], "metadata": {}}],
            "gap_papers": [{"gap_id": "gap_0", "paper_id": "p1", "rank": 0, "gap_score": 0.5}],
            "llm_analyses": [],
        }
        self.store.publish_snapshot(payload)
        with self.assertRaises(ValueError):
            self.store.build_evidence_pack(
                {
                    "snapshot_id": "snap_cue_required",
                    "target_type": "gap",
                    "gap_id": "gap_0",
                    "discovery_cue": {"text": "focus on cue usage"},
                }
            )

    def test_build_evidence_pack_rejects_non_qwen_cue_source_snapshot(self) -> None:
        target_payload = {
            "snapshot_id": "snap_target_non_qwen",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test", "split_role": "historical", "cutoff_date": "2020-12-31", "embedding_source": "qwen"},
            "papers": [
                {
                    "paper_id": "p1",
                    "title": "Paper 1",
                    "abstract": "Target paper",
                    "publication_year": 2020,
                    "cluster_id": 1,
                    "embedding": [1.0, 0.0],
                }
            ],
            "clusters": [{"cluster_id": 1, "size": 1, "metadata": {}}],
            "gaps": [{"gap_id": "gap_0", "region_index": 0, "size": 1, "avg_gap_score": 0.5, "max_gap_score": 0.5, "cluster_ids": [1], "metadata": {}}],
            "gap_papers": [{"gap_id": "gap_0", "paper_id": "p1", "rank": 0, "gap_score": 0.5}],
            "llm_analyses": [],
        }
        source_payload = {
            "snapshot_id": "snap_source_bert",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test", "split_role": "full", "embedding_source": "bert"},
            "papers": [
                {
                    "paper_id": "s1",
                    "title": "Source paper",
                    "abstract": "Source abstract",
                    "publication_year": 2020,
                    "cluster_id": 2,
                    "embedding": [1.0, 0.0],
                }
            ],
            "clusters": [{"cluster_id": 2, "size": 1, "metadata": {}}],
            "gaps": [],
            "gap_papers": [],
            "llm_analyses": [],
        }
        self.store.publish_snapshot(target_payload)
        self.store.publish_snapshot(source_payload)
        with self.assertRaises(ValueError):
            self.store.build_evidence_pack(
                {
                    "snapshot_id": "snap_target_non_qwen",
                    "target_type": "gap",
                    "gap_id": "gap_0",
                    "discovery_cue": {"text": "focus on cue usage"},
                    "cue_source_snapshot_id": "snap_source_bert",
                }
            )

    def test_build_evidence_pack_cue_similarity_sampling_is_deterministic_and_filtered(self) -> None:
        target_payload = {
            "snapshot_id": "snap_hist_target",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {
                "source": "test",
                "split_role": "historical",
                "cutoff_date": "2020-12-31",
                "embedding_source": "qwen",
            },
            "papers": [
                {
                    "paper_id": "shared",
                    "title": "Shared historical paper",
                    "abstract": "shared",
                    "publication_year": 2020,
                    "cluster_id": 1,
                    "gap_score": 0.9,
                    "embedding": [1.0, 0.0],
                }
            ],
            "clusters": [{"cluster_id": 1, "size": 1, "metadata": {}}],
            "gaps": [{"gap_id": "gap_0", "region_index": 0, "size": 1, "avg_gap_score": 0.9, "max_gap_score": 0.9, "cluster_ids": [1], "metadata": {}}],
            "gap_papers": [{"gap_id": "gap_0", "paper_id": "shared", "rank": 0, "gap_score": 0.9}],
            "llm_analyses": [],
        }
        source_payload = {
            "snapshot_id": "snap_full_source",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test", "split_role": "full", "embedding_source": "qwen"},
            "papers": [
                {"paper_id": "shared", "title": "Shared historical paper", "abstract": "shared", "publication_year": 2020, "cluster_id": 1, "embedding": [1.0, 0.0]},
                {"paper_id": "a", "title": "A", "abstract": "a", "publication_year": 2018, "cluster_id": 1, "embedding": [0.99, 0.01]},
                {"paper_id": "b", "title": "B", "abstract": "b", "publication_year": 2019, "cluster_id": 1, "embedding": [0.98, 0.02]},
                {"paper_id": "c", "title": "C", "abstract": "c", "publication_year": 2020, "cluster_id": 1, "embedding": [0.97, 0.03]},
                {"paper_id": "d", "title": "D", "abstract": "d", "publication_year": 2021, "cluster_id": 1, "embedding": [0.96, 0.04]},
                {"paper_id": "e", "title": "E", "abstract": "e", "publication_year": None, "cluster_id": 1, "embedding": [0.95, 0.05]},
                {"paper_id": "f", "title": "F", "abstract": "f", "publication_year": 2022, "cluster_id": 1, "embedding": [0.94, 0.06]},
            ],
            "clusters": [{"cluster_id": 1, "size": 7, "metadata": {}}],
            "gaps": [],
            "gap_papers": [],
            "llm_analyses": [],
        }
        self.store.publish_snapshot(target_payload)
        self.store.publish_snapshot(source_payload)
        self.store._embed_texts_with_qwen = lambda texts: [[1.0, 0.0]]  # type: ignore[method-assign]

        base_request = {
            "snapshot_id": "snap_hist_target",
            "target_type": "gap",
            "gap_id": "gap_0",
            "profile": "focused_eval",
            "exemplars": 1,
            "boundary": 1,
            "diverse": 0,
            "discovery_cue": {"text": "liposome cue"},
            "cue_source_snapshot_id": "snap_full_source",
            "cue_similarity_top_k": 7,
            "cue_similarity_sample_n": 3,
        }
        evidence_seed_1_a = self.store.build_evidence_pack({**base_request, "cue_similarity_seed": 1})
        evidence_seed_1_b = self.store.build_evidence_pack({**base_request, "cue_similarity_seed": 1})
        evidence_seed_2 = self.store.build_evidence_pack({**base_request, "cue_similarity_seed": 2})
        evidence_default_seed_a = self.store.build_evidence_pack(dict(base_request))
        evidence_default_seed_b = self.store.build_evidence_pack(dict(base_request))

        stats_1_a = dict(evidence_seed_1_a["meta"]["cue_full_similarity_stats"])
        stats_1_b = dict(evidence_seed_1_b["meta"]["cue_full_similarity_stats"])
        stats_2 = dict(evidence_seed_2["meta"]["cue_full_similarity_stats"])
        stats_default_a = dict(evidence_default_seed_a["meta"]["cue_full_similarity_stats"])
        stats_default_b = dict(evidence_default_seed_b["meta"]["cue_full_similarity_stats"])

        self.assertEqual(stats_1_a["sampled_ids"], stats_1_b["sampled_ids"])
        self.assertNotEqual(stats_1_a["sampled_ids"], stats_2["sampled_ids"])
        self.assertEqual(stats_default_a["sampled_ids"], stats_default_b["sampled_ids"])
        self.assertEqual(int(stats_1_a["filtered_by_cutoff"]), 3)
        self.assertEqual(int(stats_1_a["candidate_count"]), 4)

        sampled_years = {"shared": 2020, "a": 2018, "b": 2019, "c": 2020, "d": 2021, "e": None, "f": 2022}
        for sampled_id in stats_1_a["sampled_ids"]:
            year = sampled_years[sampled_id]
            self.assertIsNotNone(year)
            self.assertLessEqual(int(year), 2020)

        cue_similarity_papers = [
            paper
            for paper in evidence_seed_1_a["papers"]
            if "cue_full_similarity" in list(paper.get("selection_sources") or [])
        ]
        self.assertTrue(cue_similarity_papers)
        self.assertTrue(
            all(
                "cue_full_similarity_score" in dict(paper.get("selection_meta") or {})
                for paper in cue_similarity_papers
            )
        )

    def test_build_evidence_pack_includes_required_paper_ids_once(self) -> None:
        payload = {
            "snapshot_id": "snap_required_ids",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test"},
            "papers": [
                {
                    "paper_id": "p1",
                    "title": "Gap paper",
                    "abstract": "boundary evidence",
                    "publication_year": 2020,
                    "cluster_id": 1,
                    "gap_score": 0.9,
                },
                {
                    "paper_id": "p2",
                    "title": "Required paper two",
                    "abstract": "secondary evidence",
                    "publication_year": 2019,
                    "cluster_id": 1,
                    "gap_score": 0.3,
                },
                {
                    "paper_id": "p3",
                    "title": "Required paper three",
                    "abstract": "tertiary evidence",
                    "publication_year": 2018,
                    "cluster_id": 1,
                    "gap_score": 0.2,
                },
            ],
            "clusters": [{"cluster_id": 1, "size": 3, "metadata": {}}],
            "gaps": [{"gap_id": "gap_0", "region_index": 0, "size": 1, "avg_gap_score": 0.9, "max_gap_score": 0.9, "cluster_ids": [1], "metadata": {}}],
            "gap_papers": [{"gap_id": "gap_0", "paper_id": "p1", "rank": 0, "gap_score": 0.9}],
            "llm_analyses": [],
        }
        self.store.publish_snapshot(payload)

        evidence = self.store.build_evidence_pack(
            {
                "snapshot_id": "snap_required_ids",
                "target_type": "gap",
                "gap_id": "gap_0",
                "profile": "focused_eval",
                "exemplars": 1,
                "boundary": 1,
                "diverse": 0,
                "required_paper_ids": ["p3", "p3", "p2"],
            }
        )

        paper_ids = [paper["paper_id"] for paper in evidence["papers"]]
        self.assertIn("p2", paper_ids)
        self.assertIn("p3", paper_ids)
        self.assertEqual(paper_ids.count("p2"), 1)
        self.assertEqual(paper_ids.count("p3"), 1)
        self.assertEqual(evidence["meta"]["required_paper_ids"], ["p3", "p2"])
        self.assertEqual(evidence["meta"]["required_paper_source_snapshot_id"], "snap_required_ids")
        self.assertFalse(bool(evidence["meta"]["required_paper_source_is_external"]))
        self.assertEqual(int(evidence["stats"]["n_required_paper_ids"]), 2)

        required_p3 = next(paper for paper in evidence["papers"] if paper["paper_id"] == "p3")
        self.assertIn("required_paper_id", list(required_p3.get("selection_sources") or []))
        self.assertEqual(required_p3["selection_meta"]["required_paper_source_snapshot_id"], "snap_required_ids")
        self.assertFalse(bool(required_p3["selection_meta"]["required_paper_source_is_external"]))

    def test_build_evidence_pack_includes_required_paper_ids_from_source_snapshot(self) -> None:
        target_payload = {
            "snapshot_id": "snap_target_required_ids",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test"},
            "papers": [
                {
                    "paper_id": "p1",
                    "title": "Target gap paper",
                    "abstract": "target evidence",
                    "publication_year": 2020,
                    "cluster_id": 1,
                    "gap_score": 0.9,
                }
            ],
            "clusters": [{"cluster_id": 1, "size": 1, "metadata": {}}],
            "gaps": [{"gap_id": "gap_0", "region_index": 0, "size": 1, "avg_gap_score": 0.9, "max_gap_score": 0.9, "cluster_ids": [1], "metadata": {}}],
            "gap_papers": [{"gap_id": "gap_0", "paper_id": "p1", "rank": 0, "gap_score": 0.9}],
            "llm_analyses": [],
        }
        source_payload = {
            "snapshot_id": "snap_source_required_ids",
            "created_at": "2026-03-12T00:00:00+00:00",
            "metadata": {"source": "test", "split_role": "full"},
            "papers": [
                {
                    "paper_id": "p2",
                    "title": "External required paper",
                    "abstract": "external evidence",
                    "publication_year": 2021,
                    "cluster_id": 2,
                    "gap_score": 0.1,
                }
            ],
            "clusters": [{"cluster_id": 2, "size": 1, "metadata": {}}],
            "gaps": [],
            "gap_papers": [],
            "llm_analyses": [],
        }
        self.store.publish_snapshot(target_payload)
        self.store.publish_snapshot(source_payload)

        evidence = self.store.build_evidence_pack(
            {
                "snapshot_id": "snap_target_required_ids",
                "target_type": "gap",
                "gap_id": "gap_0",
                "required_paper_ids": ["p2"],
                "required_paper_source_snapshot_id": "snap_source_required_ids",
            }
        )

        paper_ids = [paper["paper_id"] for paper in evidence["papers"]]
        self.assertIn("p2", paper_ids)
        self.assertEqual(evidence["meta"]["required_paper_source_snapshot_id"], "snap_source_required_ids")
        self.assertTrue(bool(evidence["meta"]["required_paper_source_is_external"]))
        required_p2 = next(paper for paper in evidence["papers"] if paper["paper_id"] == "p2")
        self.assertEqual(required_p2["selection_meta"]["required_paper_source_snapshot_id"], "snap_source_required_ids")
        self.assertTrue(bool(required_p2["selection_meta"]["required_paper_source_is_external"]))

    def test_build_evidence_pack_rejects_unknown_required_paper_id(self) -> None:
        target_payload = {
            "snapshot_id": "snap_missing_required_id",
            "created_at": "2026-03-11T00:00:00+00:00",
            "metadata": {"source": "test"},
            "papers": [
                {
                    "paper_id": "p1",
                    "title": "Known paper",
                    "abstract": "known abstract",
                    "publication_year": 2020,
                    "cluster_id": 1,
                    "gap_score": 0.9,
                }
            ],
            "clusters": [{"cluster_id": 1, "size": 1, "metadata": {}}],
            "gaps": [{"gap_id": "gap_0", "region_index": 0, "size": 1, "avg_gap_score": 0.9, "max_gap_score": 0.9, "cluster_ids": [1], "metadata": {}}],
            "gap_papers": [{"gap_id": "gap_0", "paper_id": "p1", "rank": 0, "gap_score": 0.9}],
            "llm_analyses": [],
        }
        source_payload = {
            "snapshot_id": "snap_missing_required_source",
            "created_at": "2026-03-12T00:00:00+00:00",
            "metadata": {"source": "test", "split_role": "full"},
            "papers": [],
            "clusters": [],
            "gaps": [],
            "gap_papers": [],
            "llm_analyses": [],
        }
        self.store.publish_snapshot(target_payload)
        self.store.publish_snapshot(source_payload)

        with self.assertRaisesRegex(ValueError, "snap_missing_required_source"):
            self.store.build_evidence_pack(
                {
                    "snapshot_id": "snap_missing_required_id",
                    "target_type": "gap",
                    "gap_id": "gap_0",
                    "required_paper_ids": ["missing_paper"],
                    "required_paper_source_snapshot_id": "snap_missing_required_source",
                }
            )


if __name__ == "__main__":
    unittest.main()

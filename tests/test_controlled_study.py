from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from novelty_app.evaluation.controlled_study import (
    CONDITIONS, DIMENSIONS, BudgetExceeded, Candidates, ModelCaller, StudyConfig,
    SynthesisAssessment, assess, assessment_input, choose, digest, freeze_papers,
    generate, supported, validate_assessment, verify_task, write_json,
)
from novelty_app.evaluation.controlled_sources import fused_candidates, historical_pairs
from novelty_app.evaluation.controlled_reports import paired_bootstrap
from novelty_app.evaluation.run_controlled_study import (
    assess_stage, generate_stage, load_manifest, validate_lock,
)


def task():
    papers = freeze_papers([
        {"paper_id": "a", "title": "A", "abstract": "Premise A", "cluster_id": 1},
        {"paper_id": "b", "title": "B", "abstract": "Premise B", "cluster_id": 2}])
    t = {"task_id": "t", "domain": "test", "cue": {"text": "Question"},
         "target": {"cluster_a": 1, "cluster_b": 2}, "papers": papers, "pack_hash": digest(papers)}
    t["task_hash"] = digest(t)
    return t


def score():
    return {d: {"score": 2, "insufficient_evidence": False,
                "source_ids": ["b" if d == "premise_b" else "a"], "rationale": "Supported"}
            for d in DIMENSIONS}


def candidate(i="c"):
    return {"candidate_id": i, "position": 0, "title": "Proposal", "text": "Test A with B", "support_citations": ["a", "b"]}


class FakeCaller:
    def __init__(self):
        self.requests = []

    def call(self, role, identity, system, payload, schema, validator=None):
        self.requests.append((role, identity, payload))
        if schema is Candidates:
            out = {"hypotheses": [{k: v for k, v in candidate(str(i)).items() if k not in ("candidate_id", "position")}
                                  for i in range(3)]}
        elif schema is SynthesisAssessment:
            out = score()
        else:
            out = {"text": "Intermediate grounded text"}
        return validator(out) if validator else out


def test_pack_and_task_tampering_detected():
    t = task()
    verify_task(t)
    t["papers"][0]["abstract"] += " altered"
    with pytest.raises(ValueError, match="evidence"):
        verify_task(t)
    t = task()
    t["cue"]["text"] = "altered"
    with pytest.raises(ValueError, match="metadata"):
        verify_task(t)


def test_duplicate_sources_rejected():
    with pytest.raises(ValueError):
        freeze_papers([{"paper_id": "a"}, {"paper_id": "a"}])


def test_generation_replays_identical_evidence_and_three_candidates():
    caller = FakeCaller()
    t = task()
    ids = []
    for condition in CONDITIONS:
        for replicate in range(StudyConfig().replicates):
            output = generate(t, condition, replicate, caller)
            assert len(output) == 3
            ids.extend(c["candidate_id"] for c in output)
    assert StudyConfig().replicates == 3
    assert len(set(ids)) == 27
    assert len(caller.requests) == 15
    assert all(p["papers"] == t["papers"] for _, _, p in caller.requests)


def test_masking_and_source_validation():
    c = {**candidate(), "method_name": "secret", "future_match": "secret", "audit": "secret"}
    payload = assessment_input(task(), c)
    assert "secret" not in str(payload)
    s = score()
    assert supported(validate_assessment(s, task()))
    s["premise_a"]["source_ids"] = ["missing"]
    with pytest.raises(ValueError, match="outside"):
        validate_assessment(s, task())
    s = score()
    s["premise_b"]["source_ids"] = ["a"]
    with pytest.raises(ValueError, match="community"):
        validate_assessment(s, task())


def test_selection_ties_abstention_and_future_invariance():
    cs = [candidate("first"), candidate("second")]
    assert choose(cs, {}) is None
    scores = {c["candidate_id"]: score() for c in cs}
    assert choose(cs, scores) == "first"
    cs[1]["gold_reciprocal_rank"] = 1
    assert choose(cs, scores) == "first"
    scores["first"]["premise_a"]["score"] = None
    assert choose(cs, scores) == "second"


def test_historical_pairs_no_arbitrary_fallback():
    store = SimpleNamespace(top_gaps=lambda sid, k: [
        {"gap_id": "g1", "cluster_ids": [2, 1, -1]}, {"gap_id": "g2", "cluster_ids": [1, 2, 3]}])
    pairs = historical_pairs(store, "historical")
    assert pairs == [(1, 2, ["g1", "g2"]), (1, 3, ["g2"]), (2, 3, ["g2"])]


def test_fusion_includes_both_retrievers():
    result = fused_candidates([1, 2], [3, 2])
    assert result[0][0] == 2
    assert {i for i, _ in result} == {1, 2, 3}


def test_bootstrap_pairs_and_domains():
    result = paired_bootstrap({"a": [1, 1, 1], "b": [-1]}, 100, 42)
    assert result["difference"] == 0
    assert result["ci95"] == [0, 0]
    assert paired_bootstrap({})["difference"] is None


def test_stages_resume_and_freeze(tmp_path):
    config = StudyConfig()
    manifest = {"version": "controlled-study-v1", "config": asdict(config), "tasks": [task()]}
    manifest["manifest_hash"] = digest(manifest)
    write_json(tmp_path / "manifest.json", manifest)
    assert load_manifest(tmp_path) == manifest
    caller = FakeCaller()
    generate_stage(tmp_path, manifest, config, caller)
    count = len(caller.requests)
    generate_stage(tmp_path, manifest, config, caller)
    assert len(caller.requests) == count
    assess_stage(tmp_path, manifest, config, caller)
    validate_lock(tmp_path, manifest)
    path = next((tmp_path / "generation").glob("*.json"))
    write_json(path, {"altered": True})
    with pytest.raises(ValueError, match="changed"):
        validate_lock(tmp_path, manifest)


def test_model_caller_budget_cache_and_configuration(tmp_path, monkeypatch):
    import langchain_openai
    options = []
    class FakeLLM:
        def __init__(self, **kwargs):
            options.append(kwargs)
        def with_structured_output(self, schema, **kwargs):
            return self
        def invoke(self, messages):
            return {"parsed": Candidates(hypotheses=[]), "raw": SimpleNamespace(usage_metadata={"total_tokens": 5}, response_metadata={"model_name": "gpt-5.6-luna"})}
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeLLM)
    caller = ModelCaller(tmp_path, StudyConfig(), 1, 100000)
    caller.call("generation", [1], "system", {}, Candidates)
    caller.call("generation", [1], "system", {}, Candidates)
    assert caller.calls == 1
    assert options[0]["model"] == "gpt-5.6-luna"
    assert options[0]["reasoning_effort"] == "medium"
    with pytest.raises(BudgetExceeded):
        caller.call("generation", [2], "system", {}, Candidates)


def test_read_only_store_cannot_write(tmp_path):
    import sqlite3
    from novelty_app.evaluation.controlled_sources import ReadOnlyStore
    path = tmp_path / "archive.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE original (id INTEGER)")
    original = path.read_bytes()
    store = ReadOnlyStore(path)
    with store._connect() as conn:
        assert conn.execute("SELECT count(*) FROM original").fetchone()[0] == 0
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO original VALUES (1)")
    assert path.read_bytes() == original


def test_complete_report_from_mocked_study(tmp_path):
    from novelty_app.evaluation.controlled_reports import analyse
    config = StudyConfig(bootstrap_samples=100)
    manifest = {"version": "controlled-study-v1", "config": asdict(config), "tasks": [task()]}
    manifest["manifest_hash"] = digest(manifest)
    write_json(tmp_path / "manifest.json", manifest)
    caller = FakeCaller()
    generate_stage(tmp_path, manifest, config, caller)
    assess_stage(tmp_path, manifest, config, caller)
    analyse(tmp_path, manifest, config)
    from novelty_app.evaluation.controlled_study import read_json
    report = read_json(tmp_path / "reports" / "analysis.json")
    assert all(s["generated"] == 9 for s in report["summary"].values())
    assert all(s["supported_rate"] == 1 for s in report["summary"].values())
    assert len(report["correspondence_by_selection_policy"]) == 3
    for result in report["correspondence_by_selection_policy"].values():
        assert result["eligible_candidates"] == 3
        assert result["completely_matched"] == 0
        assert result["shared_proposal"] is None
    assert (tmp_path / "reports" / "synthesis_rates.pdf").exists()


def test_later_retrieval_does_not_inject_gold_and_preserves_semantic_candidates():
    import pandas as pd
    from novelty_app.evaluation.candidate_match import build_corpus_index
    from novelty_app.evaluation.controlled_sources import retrieve_later
    df = pd.DataFrame([{"paper_id": "p1", "title": "A", "abstract": "material", "publication_year": 2021},
                       {"paper_id": "p2", "title": "B", "abstract": "other topic", "publication_year": 2022}])
    corpus = build_corpus_index(df, np.eye(2))
    class Qwen:
        def embed(self, *args, **kwargs):
            return [[0., 1.]]
        def rank(self, *, query, documents, top_k, require_reranker):
            assert require_reranker
            assert len(documents) == 2
            return [{"index": i, "reranker_score": 0.9 - i/10} for i in range(2)]
    results = retrieve_later(candidate(), corpus, Qwen())
    assert len(results) == 2
    assert all("fusion_score" in r for r in results)


def test_qwen_strict_mode_rejects_fallback(monkeypatch):
    from novelty_app.evaluation.qwen_client import QwenClient
    qwen = QwenClient()
    monkeypatch.setattr(qwen, "_post", lambda *args: {"results": [], "used_embedding_fallback": True})
    with pytest.raises(RuntimeError, match="Strict"):
        qwen.rank(query="q", documents=["d"], require_reranker=True)
    assert qwen.rank(query="q", documents=["d"]) == []
def test_stored_match_sensitivity():
    from novelty_app.evaluation.controlled_reports import stored_match_sensitivity
    from novelty_app.evaluation.judge import judge_candidate_match
    fingerprint = {"material": ["liposome"]}
    candidate = {"title": "liposome", "abstract": "liposome", "reranker_score": .39, "embedding_score": .5}
    candidate["judge"] = judge_candidate_match(fingerprint, candidate)
    result = stored_match_sensitivity([{"method_name": "direct", "hypothesis": {"idea_fingerprint": fingerprint},
                                        "historical_match": candidate}])
    group = result["groups"]["direct:historical_match"]
    assert group["available"] == 1
    assert group["baseline_disagreements"] == 0
    assert len(group["label_changes"]) == 22
    assert result["groups"]["direct:future_match"]["missing"] == 1


def test_matching_sample_balanced_and_outcome_independent():
    from collections import Counter
    from novelty_app.evaluation.controlled_matching import matching_sample
    tasks = [{"task_id": f"{d}-{i}", "domain": d} for d in "abcd" for i in range(50)]
    sample = matching_sample({"tasks": tasks})
    assert len(sample) == len(set(sample)) == 100
    assert Counter(t.split("-")[0] for t in sample) == dict.fromkeys("abcd", 25)
    assert sample == matching_sample({"tasks": [{**t, "future_result": True} for t in reversed(tasks)]})
    assert len(matching_sample({"tasks": tasks[:4]})) == 4


def test_joint_matching_validates_each_publication():
    from novelty_app.evaluation.controlled_matching import assess_publications
    papers = [{"paper_id": str(i), "rank": i + 1, "title": "Title", "abstract": f"Finding {i}."} for i in range(10)]
    judgments = [{"paper_id": p["paper_id"], "correspondence": "shared_proposal", "outcome": "supporting",
                  "source_passages": [p["abstract"]], "prediction": "Test", "rationale": "Evidence"} for p in papers]
    class JointCaller:
        calls = 0
        def call(self, role, identity, system, payload, schema, validate):
            self.calls += 1
            assert len(payload["publications"]) == 10
            return validate(schema.model_validate({"publications": judgments}).model_dump())
    caller = JointCaller()
    assert len(assess_publications(candidate(), papers, caller, ["test"])) == 10
    assert caller.calls == 1
    judgments[0]["source_passages"] = [papers[1]["abstract"]]
    with pytest.raises(ValueError, match="Quote absent"):
        assess_publications(candidate(), papers, caller, ["test"])
    judgments[0]["source_passages"] = [papers[0]["abstract"]]
    judgments[-1]["paper_id"] = "0"
    with pytest.raises(ValueError, match="exactly once"):
        assess_publications(candidate(), papers, caller, ["test"])

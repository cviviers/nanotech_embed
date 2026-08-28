from __future__ import annotations

import unittest
from unittest.mock import patch

from novelty_app.agents.orchestrator_langgraph import (
    ContrastiveExplanation,
    Hypothesis,
    HypothesesOut,
    InstrumentedOrchestrator,
    node_explain,
    node_publish,
)
from novelty_app.evaluation.generators import (
    GenerationContext,
    generate_cue_retrieval_generation,
    generate_single_shot_llm,
)
from novelty_app.evaluation.judge import (
    CriterionScore,
    HypothesisIdeaScore,
    HypothesisIdeaScoresOut,
    score_hypotheses,
)


class _FakeBackend:
    def __init__(self) -> None:
        self.artifact_calls = []
        self.evidence_pack_calls = []

    def evidence_pack(self, payload):
        self.evidence_pack_calls.append(dict(payload))
        return {
            "snapshot_id": payload.get("snapshot_id"),
            "target_type": payload.get("target_type"),
            "papers": [
                {
                    "paper_id": "p1",
                    "title": "Targeted liposome delivery",
                    "abstract": "liposome delivery for cancer",
                    "publication_year": 2020,
                }
            ],
            "meta": {"profile": payload.get("profile", "focused_eval")},
        }

    def store_artifact(self, *, kind, target, payload):
        self.artifact_calls.append({"kind": kind, "target": target, "payload": payload})
        return {"artifact_id": "artifact_1", "kind": kind}


class _FakeStructuredInvoker:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append({"messages": messages, "config": config})
        return self.response


class _FakeChatOpenAI:
    last_instance = None
    next_structured = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.structured = _FakeChatOpenAI.next_structured
        self.invoke_calls = []
        _FakeChatOpenAI.last_instance = self

    def with_structured_output(self, _schema, method=None):
        if method == "function_calling":
            if self.structured is None:
                raise AssertionError("structured response must be preset in test")
            return self.structured
        raise AssertionError(f"unexpected method: {method}")

    def invoke(self, messages, config=None):
        self.invoke_calls.append({"messages": messages, "config": config})
        return type("Resp", (), {"content": "summary"})()


class _FakeCompiledGraph:
    def __init__(self):
        self.calls = []

    def invoke(self, state, *args, **kwargs):
        self.calls.append({"state": state, "args": args, "kwargs": kwargs})
        return {"published": True, "iter": 1, "evidence": [], "published_artifact": {"artifact_id": "a1"}}


class _StrictCallbackHandler:
    def __init__(self, *, session_id=None, tags=None, enabled=None):
        self.session_id = session_id
        self.tags = tags
        self.enabled = enabled


class LangchainTracingTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeChatOpenAI.last_instance = None
        _FakeChatOpenAI.next_structured = None

    def test_langchain_config_helper_appends_callback(self) -> None:
        from novelty_app.agents import observability

        with patch.object(observability, "LangfuseLangchainCallbackHandler", _StrictCallbackHandler), patch(
            "novelty_app.agents.observability.get_langfuse_client",
            return_value=object(),
        ):
            config = observability.langchain_config_with_observability(
                {"callbacks": ["existing"]},
                session_id="session_1",
                tags=["judge"],
                trace_name="ignored_by_strict_handler",
                metadata={"ignored": "value"},
            )

        self.assertIsNotNone(config)
        self.assertEqual(config["callbacks"][0], "existing")
        self.assertIsInstance(config["callbacks"][1], _StrictCallbackHandler)
        self.assertEqual(config["callbacks"][1].session_id, "session_1")
        self.assertEqual(config["callbacks"][1].tags, ["judge"])

    def test_generate_single_shot_llm_passes_langfuse_config_to_structured_invoke(self) -> None:
        backend = _FakeBackend()
        context = GenerationContext(
            backend=backend,
            snapshot_id="snapshot_1",
            target={"target_type": "gap", "gap_id": "gap_1"},
            openai_api_key="test-key",
            model_name="test-model",
            hypotheses_per_target=1,
        )
        sentinel_config = {"callbacks": ["langfuse-handler"]}
        _FakeChatOpenAI.last_instance = None
        fake_response = HypothesesOut(
            hypotheses=[
                Hypothesis(
                    id="hyp_1",
                    title="Folate liposome bridge",
                    bridge_type="delivery",
                    mechanistic_rationale="Use folate-targeted liposomes for siRNA delivery.",
                    citations=["p1"],
                )
            ]
        )
        fake_structured = _FakeStructuredInvoker(fake_response)
        _FakeChatOpenAI.next_structured = fake_structured

        with patch("novelty_app.evaluation.generators.ChatOpenAI", _FakeChatOpenAI), patch(
            "novelty_app.evaluation.generators.langchain_config_with_observability",
            return_value=sentinel_config,
        ):
            generated, _meta = generate_single_shot_llm(context)

        self.assertEqual(len(generated), 1)
        instance = _FakeChatOpenAI.last_instance
        self.assertIsNotNone(instance)
        self.assertIs(instance.structured, fake_structured)
        self.assertEqual(fake_structured.calls[0]["config"], sentinel_config)

    def test_generate_single_shot_llm_forwards_required_paper_ids_to_evidence_pack(self) -> None:
        backend = _FakeBackend()
        context = GenerationContext(
            backend=backend,
            snapshot_id="snapshot_1",
            target={"target_type": "gap", "gap_id": "gap_1"},
            openai_api_key="test-key",
            model_name="test-model",
            hypotheses_per_target=1,
            required_paper_ids=["paper_7", "paper_8"],
            required_paper_source_snapshot_id="snapshot_full_77",
        )
        fake_response = HypothesesOut(
            hypotheses=[
                Hypothesis(
                    id="hyp_1",
                    title="Folate liposome bridge",
                    bridge_type="delivery",
                    mechanistic_rationale="Use folate-targeted liposomes for siRNA delivery.",
                    citations=["p1"],
                )
            ]
        )
        _FakeChatOpenAI.next_structured = _FakeStructuredInvoker(fake_response)

        with patch("novelty_app.evaluation.generators.ChatOpenAI", _FakeChatOpenAI):
            generate_single_shot_llm(context)

        self.assertEqual(backend.evidence_pack_calls[0]["required_paper_ids"], ["paper_7", "paper_8"])
        self.assertEqual(
            backend.evidence_pack_calls[0]["required_paper_source_snapshot_id"],
            "snapshot_full_77",
        )

    def test_cue_retrieval_generation_does_not_send_frontier_target_to_retrieval(self) -> None:
        backend = _FakeBackend()
        context = GenerationContext(
            backend=backend,
            snapshot_id="snapshot_historical",
            target={"target_type": "gap", "gap_id": "secret_gap"},
            openai_api_key="test-key",
            model_name="test-model",
            discovery_cue={"text": "Find a nanoparticle system that detects DNA and RNA"},
            cue_source_snapshot_id="snapshot_historical",
            cue_similarity_top_k=20,
            cue_similarity_sample_n=6,
            cue_similarity_seed="seed_1",
            hypotheses_per_target=1,
        )
        fake_response = HypothesesOut(
            hypotheses=[
                Hypothesis(
                    id="hyp_1",
                    title="Dual nucleic-acid nanosensor",
                    bridge_type="biosensing",
                    mechanistic_rationale="Use orthogonal probes for DNA and RNA detection.",
                    citations=["p1"],
                )
            ]
        )
        fake_structured = _FakeStructuredInvoker(fake_response)
        _FakeChatOpenAI.next_structured = fake_structured

        with patch("novelty_app.evaluation.generators.ChatOpenAI", _FakeChatOpenAI):
            generated, meta = generate_cue_retrieval_generation(context)

        request = backend.evidence_pack_calls[0]
        self.assertEqual(request["target_type"], "cue")
        self.assertNotIn("gap_id", request)
        self.assertNotIn("cluster_a", request)
        self.assertNotIn("cluster_b", request)
        self.assertNotIn("cue_similarity_seed", request)
        self.assertEqual(request["cue_source_snapshot_id"], "snapshot_historical")
        self.assertEqual(request["exemplars"], 0)
        self.assertEqual(request["boundary"], 0)
        self.assertEqual(generated[0].target_id, "secret_gap")
        self.assertEqual(generated[0].method_name, "cue_retrieval_generation")
        self.assertEqual(meta["evidence_pack"]["target_type"], "cue")
        prompt = fake_structured.calls[0]["messages"][1]["content"]
        self.assertNotIn("secret_gap", prompt)

    def test_node_explain_passes_langfuse_config_to_structured_invoke(self) -> None:
        llm = _FakeChatOpenAI(model="test-model")
        sentinel_config = {"callbacks": ["langfuse-handler"]}
        fake_structured = _FakeStructuredInvoker(
            ContrastiveExplanation(
                cluster_A_summary={"one_line": "A", "bullets": [], "salient_entities": {}, "citations": []},
                cluster_B_summary={"one_line": "B", "bullets": [], "salient_entities": {}, "citations": []},
                axes_of_separation=[],
                bridge_seeds=[],
            )
        )
        llm.structured = fake_structured
        state = {
            "target_type": "gap",
            "gap_id": "gap_1",
            "snapshot_id": "snapshot_1",
            "evidence": [{"paper_id": "p1", "title": "Paper 1", "abstract": "Text"}],
            "discovery_cue": {"text": "focus"},
        }

        with patch(
            "novelty_app.agents.orchestrator_langgraph.langchain_config_with_observability",
            return_value=sentinel_config,
        ):
            out = node_explain(state, llm)

        self.assertIn("explanation", out)
        self.assertEqual(fake_structured.calls[0]["config"], sentinel_config)

    def test_instrumented_orchestrator_adds_langfuse_config_to_graph_invoke(self) -> None:
        compiled = _FakeCompiledGraph()
        app = InstrumentedOrchestrator(compiled)
        sentinel_config = {"callbacks": ["langfuse-handler"]}

        with patch(
            "novelty_app.agents.orchestrator_langgraph.langchain_config_with_observability",
            return_value=sentinel_config,
        ):
            out = app.invoke({"snapshot_id": "snapshot_1", "target_type": "gap", "gap_id": "gap_1"})

        self.assertTrue(out["published"])
        self.assertEqual(compiled.calls[0]["kwargs"]["config"], sentinel_config)

    def test_node_publish_persists_exact_evidence_payload(self) -> None:
        backend = _FakeBackend()
        state = {
            "target_type": "gap",
            "gap_id": "gap_1",
            "snapshot_id": "snapshot_1",
            "evidence": [{"paper_id": "p1", "title": "Paper 1", "abstract": "Text"}],
            "evidence_meta": {"profile": "focused_eval"},
            "discovery_cue": {"text": "focus"},
            "explanation": {"cluster_A_summary": {"one_line": "A"}},
            "audit": {"supported_claim_fraction": 1.0},
            "hypotheses": {"hypotheses": [{"id": "h1"}]},
            "idea_scores": {"h1": {"average_score": 4.0}},
            "blueprint": {"bill_of_materials": ["liposome"]},
            "iter": 1,
        }

        out = node_publish(state, backend)

        self.assertTrue(out["published"])
        self.assertEqual(len(backend.artifact_calls), 1)
        self.assertEqual(backend.artifact_calls[0]["payload"]["evidence_size"], 1)
        self.assertEqual(backend.artifact_calls[0]["payload"]["evidence"][0]["paper_id"], "p1")

    def test_score_hypotheses_passes_langfuse_config_to_structured_invoke(self) -> None:
        sentinel_config = {"callbacks": ["langfuse-handler"]}
        fake_response = HypothesisIdeaScoresOut(
            scored_hypotheses=[
                HypothesisIdeaScore(
                    hypothesis_id="hyp_1",
                    importance=CriterionScore(score=4, rationale="important"),
                    novelty=CriterionScore(score=4, rationale="novel"),
                    plausibility=CriterionScore(score=4, rationale="plausible"),
                    feasibility=CriterionScore(score=4, rationale="feasible"),
                    evaluability=CriterionScore(score=4, rationale="evaluable"),
                    likely_impact=CriterionScore(score=4, rationale="impactful"),
                    average_score=4.0,
                    summary="good",
                )
            ]
        )
        fake_structured = _FakeStructuredInvoker(fake_response)
        _FakeChatOpenAI.last_instance = None
        _FakeChatOpenAI.next_structured = fake_structured

        with patch("novelty_app.evaluation.judge.ChatOpenAI", _FakeChatOpenAI), patch(
            "novelty_app.evaluation.judge.langchain_config_with_observability",
            return_value=sentinel_config,
        ):
            scored = score_hypotheses(
                [{"hypothesis_id": "hyp_1", "title": "Idea", "text": "Text", "support_citations": ["p1"]}],
                evidence_pack={"papers": [{"paper_id": "p1", "title": "Paper", "abstract": "Text"}]},
                target={"snapshot_id": "snapshot_1", "target_type": "gap", "gap_id": "gap_1"},
                discovery_cue={"text": "focus"},
                openai_api_key="test-key",
                model_name="judge-model",
            )

        self.assertIn("hyp_1", scored)
        self.assertEqual(fake_structured.calls[0]["config"], sentinel_config)


if __name__ == "__main__":
    unittest.main()

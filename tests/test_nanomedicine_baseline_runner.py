from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from novelty_app.evaluation import run_nanomedicine_baselines as runner


def _matching_review_packet(step: runner.RunStep) -> dict:
    return {
        "run": {
            "status": "completed",
            "snapshot_id": step.domain.snapshot_id,
            "method_names": [step.method],
            "cutoff_date": runner.PROTOCOL["cutoff_date"],
            "future_window_start": runner.PROTOCOL["future_window_start"],
            "future_window_end": runner.PROTOCOL["future_window_end"],
            "metrics": {"n_task_evaluations": runner.PROTOCOL["n_gold_future_papers"]},
            "config": {
                "seeds": runner.PROTOCOL["seeds"],
                "hypotheses_per_target": runner.PROTOCOL["hypotheses_per_target"],
                "n_gold_future_papers": runner.PROTOCOL["n_gold_future_papers"],
                "disable_leakage_check": True,
                "cue_source_snapshot_id": step.domain.snapshot_id,
                "future_prefilter": {
                    "semantic_query": step.domain.future_semantic_query,
                    "semantic_threshold": runner.PROTOCOL["future_semantic_threshold"],
                },
            },
        },
        "rows": [],
    }


class NanomedicineBaselineRunnerTests(unittest.TestCase):
    def test_default_method_list_includes_all_registered_methods_including_orchestrator(self) -> None:
        self.assertEqual(runner.DEFAULT_METHODS, runner.REGISTERED_METHODS)
        self.assertIn("orchestrator", runner.DEFAULT_METHODS)
        self.assertIn("cue_retrieval_generation", runner.DEFAULT_METHODS)

    def test_explicit_methods_limit_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.parse_args(
                [
                    "--qwen-base-url",
                    "http://192.168.2.35:800",
                    "--domains",
                    "payload",
                    "vaccine",
                    "--methods",
                    "pack_query_baseline",
                    "heuristic_bridge",
                    "--output-root",
                    tmpdir,
                ]
            )
            steps = runner.build_steps(args)

        self.assertEqual(
            [(step.domain.name, step.method) for step in steps],
            [
                ("payload", "pack_query_baseline"),
                ("payload", "heuristic_bridge"),
                ("vaccine", "pack_query_baseline"),
                ("vaccine", "heuristic_bridge"),
            ],
        )

    def test_dry_run_plans_four_domains_by_all_default_methods_without_subprocesses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            runner,
            "validate_runtime_inputs",
            return_value=None,
        ), patch.object(runner.subprocess, "Popen") as popen_mock, patch.object(
            runner.subprocess,
            "run",
        ) as run_mock:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = runner.main(
                    [
                        "--dry-run",
                        "--qwen-base-url",
                        "http://192.168.2.35:800",
                        "--output-root",
                        tmpdir,
                    ]
                )

        self.assertEqual(code, 0)
        self.assertIn(f"Dry run: {4 * len(runner.DEFAULT_METHODS)} step(s) planned.", stdout.getvalue())
        self.assertFalse(popen_mock.called)
        self.assertFalse(run_mock.called)

    def test_domain_configs_capture_archived_snapshot_ids_and_queries(self) -> None:
        self.assertEqual(
            runner.DOMAIN_CONFIGS["antimicrobials"].snapshot_id,
            "snapshot_5ea0197501_historical",
        )
        self.assertEqual(
            runner.DOMAIN_CONFIGS["payload"].snapshot_id,
            "snapshot_f21bfc36ed_historical",
        )
        self.assertIn("biofilm", runner.DOMAIN_CONFIGS["antimicrobials"].future_semantic_query.lower())
        self.assertIn("protein payloads", runner.DOMAIN_CONFIGS["payload"].discovery_cue_text.lower())

    def test_completed_output_detection_skips_matching_review_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            step = runner.RunStep(
                domain=runner.DOMAIN_CONFIGS["payload"],
                method="pack_query_baseline",
                output_dir=Path(tmpdir),
            )
            packet_path = step.output_dir / "retro_eval_test_review_packet.json"
            packet_path.write_text(json.dumps(_matching_review_packet(step)), encoding="utf-8")

            self.assertEqual(runner.completed_review_packet(step), packet_path)

    def test_mismatched_review_packet_is_not_considered_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            step = runner.RunStep(
                domain=runner.DOMAIN_CONFIGS["payload"],
                method="pack_query_baseline",
                output_dir=Path(tmpdir),
            )
            payload = _matching_review_packet(step)
            payload["run"]["method_names"] = ["heuristic_bridge"]
            packet_path = step.output_dir / "retro_eval_test_review_packet.json"
            packet_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertIsNone(runner.completed_review_packet(step))

    def test_zero_task_review_packet_is_not_considered_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            step = runner.RunStep(
                domain=runner.DOMAIN_CONFIGS["payload"],
                method="cue_retrieval_generation",
                output_dir=Path(tmpdir),
            )
            payload = _matching_review_packet(step)
            payload["run"]["metrics"]["n_task_evaluations"] = 0
            packet_path = step.output_dir / "retro_eval_failed_review_packet.json"
            packet_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertIsNone(runner.completed_review_packet(step))

    def test_retrospective_command_includes_cue_source_snapshot_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.parse_args(
                [
                    "--qwen-base-url",
                    "http://192.168.2.35:800",
                    "--domains",
                    "antimicrobials",
                    "--methods",
                    "pack_query_baseline",
                    "--output-root",
                    tmpdir,
                ]
            )
            step = runner.build_steps(args)[0]

        command = runner.build_retrospective_command(step, args)
        self.assertIn("--cue-source-snapshot-id", command)
        self.assertEqual(
            command[command.index("--cue-source-snapshot-id") + 1],
            step.domain.snapshot_id,
        )

    def test_retrospective_command_includes_default_benchmark_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.parse_args(
                [
                    "--qwen-base-url",
                    "http://192.168.2.35:800",
                    "--domains",
                    "payload",
                    "--methods",
                    "pack_query_baseline",
                    "--output-root",
                    tmpdir,
                ]
            )
            step = runner.build_steps(args)[0]

        command = runner.build_retrospective_command(step, args)
        self.assertIn("--benchmark-cache-path", command)
        cache_path = Path(command[command.index("--benchmark-cache-path") + 1])
        self.assertEqual(cache_path, Path(tmpdir) / "payload" / "_benchmark_cache" / "benchmark.json")

    def test_disable_benchmark_cache_omits_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.parse_args(
                [
                    "--qwen-base-url",
                    "http://192.168.2.35:800",
                    "--domains",
                    "payload",
                    "--methods",
                    "pack_query_baseline",
                    "--output-root",
                    tmpdir,
                    "--disable-benchmark-cache",
                ]
            )
            step = runner.build_steps(args)[0]

        command = runner.build_retrospective_command(step, args)
        self.assertNotIn("--benchmark-cache-path", command)

    def test_methods_for_one_domain_share_benchmark_cache_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.parse_args(
                [
                    "--qwen-base-url",
                    "http://192.168.2.35:800",
                    "--domains",
                    "payload",
                    "--methods",
                    "pack_query_baseline",
                    "heuristic_bridge",
                    "--output-root",
                    tmpdir,
                ]
            )
            steps = runner.build_steps(args)

        cache_paths = {runner.benchmark_cache_path_for_step(step, args) for step in steps}
        self.assertEqual(len(cache_paths), 1)
        self.assertEqual(next(iter(cache_paths)), Path(tmpdir) / "payload" / "_benchmark_cache" / "benchmark.json")

    def test_subprocess_env_sets_qwen_base_url_and_domain_db(self) -> None:
        args = runner.parse_args(
            [
                "--qwen-base-url",
                "http://192.168.2.35:800/",
                "--domains",
                "antimicrobials",
                "--methods",
                "pack_query_baseline",
            ]
        )

        env = runner._subprocess_env(args, domain=runner.DOMAIN_CONFIGS["antimicrobials"])

        self.assertEqual(env["QWEN_BASE_URL"], "http://192.168.2.35:800")
        self.assertEqual(
            env["NOVELTY_AGENT_DB"],
            str(runner.DOMAIN_CONFIGS["antimicrobials"].archived_db.resolve()),
        )

    def test_health_qwen_matching_normalizes_trailing_slash(self) -> None:
        self.assertTrue(
            runner._health_qwen_matches(
                {"qwen_base_url": "http://192.168.2.35:800/"},
                "http://192.168.2.35:800",
            )
        )

    def test_run_all_skips_completed_step_without_starting_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            runner,
            "validate_runtime_inputs",
            return_value=None,
        ), patch.object(runner, "ManagedBackend") as backend_mock, patch.object(
            runner,
            "_run_step",
        ) as run_step_mock:
            args = runner.parse_args(
                [
                    "--qwen-base-url",
                    "http://192.168.2.35:800",
                    "--domains",
                    "payload",
                    "--methods",
                    "pack_query_baseline",
                    "--output-root",
                    tmpdir,
                ]
            )
            step = runner.build_steps(args)[0]
            step.output_dir.mkdir(parents=True)
            (step.output_dir / "retro_eval_done_review_packet.json").write_text(
                json.dumps(_matching_review_packet(step)),
                encoding="utf-8",
            )

            code = runner.run_all(args)

        self.assertEqual(code, 0)
        self.assertFalse(backend_mock.called)
        self.assertFalse(run_step_mock.called)

    def test_run_all_reruns_incomplete_step_with_backend_context(self) -> None:
        backend = MagicMock()
        backend.__enter__.return_value = backend
        backend.__exit__.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            runner,
            "validate_runtime_inputs",
            return_value=None,
        ), patch.object(runner, "ManagedBackend", return_value=backend) as backend_mock, patch.object(
            runner,
            "_run_step",
            return_value=0,
        ) as run_step_mock:
            args = runner.parse_args(
                [
                    "--qwen-base-url",
                    "http://192.168.2.35:800",
                    "--domains",
                    "payload",
                    "--methods",
                    "pack_query_baseline",
                    "--output-root",
                    tmpdir,
                ]
            )
            code = runner.run_all(args)

        self.assertEqual(code, 0)
        self.assertTrue(backend_mock.called)
        self.assertEqual(backend_mock.call_args.kwargs["env"]["QWEN_BASE_URL"], "http://192.168.2.35:800")
        self.assertTrue(run_step_mock.called)

    def test_openai_backed_default_methods_fail_early_without_api_key(self) -> None:
        args = runner.parse_args(["--qwen-base-url", "http://192.168.2.35:800"])
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY is required"):
                runner.validate_runtime_inputs(args, dry_run=False)

    def test_openai_backed_method_fails_early_without_langchain_openai(self) -> None:
        args = runner.parse_args(
            [
                "--qwen-base-url",
                "http://192.168.2.35:800",
                "--methods",
                "cue_retrieval_generation",
            ]
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), patch.object(
            runner,
            "_chat_openai_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(ValueError, "langchain-openai is required"):
                runner.validate_runtime_inputs(args, dry_run=False)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = PROJECT_ROOT / "data" / "nanomedicine"
ARCHIVE_DATA_DIR = ARCHIVE_ROOT / "data"
ARCHIVE_DATA_JSON = ARCHIVE_DATA_DIR / "cleaned_dataset.json"
DEFAULT_OUTPUT_ROOT = ARCHIVE_ROOT / "baseline_runs"

FALLBACK_REGISTERED_METHODS = [
    "orchestrator",
    "single_shot_llm",
    "cue_retrieval_generation",
    "retrieval_summary_direct",
    "heuristic_bridge",
    "pack_query_baseline",
    "random_target_control",
]


def _registered_methods() -> List[str]:
    try:
        from .generators import GENERATOR_REGISTRY

        return list(GENERATOR_REGISTRY)
    except Exception:
        return list(FALLBACK_REGISTERED_METHODS)


REGISTERED_METHODS = _registered_methods()
DEFAULT_METHODS = list(REGISTERED_METHODS)
OPENAI_REQUIRED_METHODS = {
    "orchestrator",
    "single_shot_llm",
    "cue_retrieval_generation",
    "retrieval_summary_direct",
}

PROTOCOL = {
    "cutoff_date": "2019-12-31",
    "future_window_start": "2020-01-01",
    "future_window_end": "2026-01-01",
    "n_gap_targets": 20,
    "n_cluster_pair_targets": 10,
    "n_gold_future_papers": 50,
    "seeds": 1,
    "hypotheses_per_target": 3,
    "future_semantic_threshold": 0.45,
}


@dataclass(frozen=True)
class DomainConfig:
    name: str
    archived_db: Path
    snapshot_id: str
    discovery_cue_text: str
    future_semantic_query: str


@dataclass(frozen=True)
class RunStep:
    domain: DomainConfig
    method: str
    output_dir: Path


DOMAIN_CONFIGS: Dict[str, DomainConfig] = {
    "antimicrobials": DomainConfig(
        name="antimicrobials",
        archived_db=ARCHIVE_ROOT
        / "retrospective_eval_antimicrobials_full_20260328"
        / "novelty_agent_knowledge.sqlite",
        snapshot_id="snapshot_5ea0197501_historical",
        discovery_cue_text="What characteristics should a coating for inorganic nanoparticles have to overcome biofilms?",
        future_semantic_query=(
            "bacteria, antimicrobial, inorganic nanoparticles, coating, biofilm, surface, infection."
        ),
    ),
    "payload": DomainConfig(
        name="payload",
        archived_db=ARCHIVE_ROOT
        / "retrospective_eval_payload_full_20260329"
        / "novelty_agent_knowledge.sqlite",
        snapshot_id="snapshot_f21bfc36ed_historical",
        discovery_cue_text=(
            "What are innovative strategies to incorporate both protein payloads and nucleic acids "
            "in the same nanoparticle platform?"
        ),
        future_semantic_query="payload, incorporation, functionalization, protein, delivery, multifunction",
    ),
    "biosensing": DomainConfig(
        name="biosensing",
        archived_db=ARCHIVE_ROOT
        / "retrospective_eval_biosensing_full_20260329"
        / "novelty_agent_knowledge.sqlite",
        snapshot_id="snapshot_6f8a2c8253_historical",
        discovery_cue_text="I'm looking for a nanoparticle biosensing system that can simultaneously detect DNA and RNA.",
        future_semantic_query=(
            "co-detection, sensing, biosensor, dual-target, DNA, RNA, nucleic acids, nanosensor."
        ),
    ),
    "vaccine": DomainConfig(
        name="vaccine",
        archived_db=ARCHIVE_ROOT
        / "retrospective_eval_vaccine_full_20260329"
        / "novelty_agent_knowledge.sqlite",
        snapshot_id="snapshot_f9b70ff871_historical",
        discovery_cue_text="What adjuvants can be included in mRNA LNP vaccines to improve their long term efficacy?",
        future_semantic_query=(
            "vaccine, adjuvants, mRNA lipid nanoparticles, ionizable lipid, antibody titer, "
            "immune memory, immunity."
        ),
    ),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _normalize_methods(methods: Sequence[str]) -> List[str]:
    valid = set(REGISTERED_METHODS)
    out: List[str] = []
    unknown = []
    for method in methods:
        text = str(method or "").strip()
        if not text:
            continue
        if text not in valid:
            unknown.append(text)
            continue
        if text not in out:
            out.append(text)
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}")
    if not out:
        raise ValueError("At least one method is required.")
    return out


def _normalize_domains(domains: Sequence[str]) -> List[str]:
    out: List[str] = []
    unknown = []
    for domain in domains:
        text = str(domain or "").strip()
        if not text:
            continue
        if text not in DOMAIN_CONFIGS:
            unknown.append(text)
            continue
        if text not in out:
            out.append(text)
    if unknown:
        raise ValueError(f"Unknown domain(s): {', '.join(unknown)}")
    if not out:
        raise ValueError("At least one domain is required.")
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run archived nanomedicine retrospective evaluations across domains and methods."
    )
    parser.add_argument("--qwen-base-url", required=True)
    parser.add_argument("--domains", nargs="+", default=list(DOMAIN_CONFIGS), choices=sorted(DOMAIN_CONFIGS))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=sorted(REGISTERED_METHODS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--backend-port", type=int, default=18088)
    parser.add_argument("--force", action="store_true", help="Rerun steps even if a matching review packet exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned steps without running them.")
    parser.add_argument(
        "--disable-benchmark-cache",
        action="store_true",
        help="Do not pass a shared benchmark cache path to retrospective evaluations.",
    )
    parser.add_argument(
        "--clear-openai-env",
        action="store_true",
        help="Remove OpenAI-related environment variables for subprocesses.",
    )
    return parser.parse_args(argv)


def build_steps(args: argparse.Namespace) -> List[RunStep]:
    methods = _normalize_methods(args.methods)
    domains = _normalize_domains(args.domains)
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    return [
        RunStep(
            domain=DOMAIN_CONFIGS[domain],
            method=method,
            output_dir=output_root / domain / method,
        )
        for domain in domains
        for method in methods
    ]


def _selected_methods_require_openai(methods: Iterable[str]) -> bool:
    return bool(OPENAI_REQUIRED_METHODS.intersection(set(methods)))


def _chat_openai_available() -> bool:
    try:
        from .generators import ChatOpenAI
    except Exception:
        return False
    return ChatOpenAI is not None


def _normalize_qwen_base_url(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def validate_runtime_inputs(args: argparse.Namespace, *, dry_run: bool = False) -> None:
    methods = _normalize_methods(args.methods)
    _normalize_domains(args.domains)

    if not ARCHIVE_DATA_JSON.exists():
        raise ValueError(f"Missing data JSON: {ARCHIVE_DATA_JSON}")
    if not ARCHIVE_DATA_DIR.exists():
        raise ValueError(f"Missing data directory: {ARCHIVE_DATA_DIR}")

    missing_dbs = [
        str(DOMAIN_CONFIGS[domain].archived_db)
        for domain in _normalize_domains(args.domains)
        if not DOMAIN_CONFIGS[domain].archived_db.exists()
    ]
    if missing_dbs:
        raise ValueError("Missing archived backend DB(s): " + "; ".join(missing_dbs))

    env = _subprocess_env(args, domain=None)
    needs_openai = _selected_methods_require_openai(methods)
    if needs_openai and not env.get("OPENAI_API_KEY") and not dry_run:
        required = ", ".join(method for method in methods if method in OPENAI_REQUIRED_METHODS)
        raise ValueError(f"OPENAI_API_KEY is required for selected method(s): {required}")
    if needs_openai and not _chat_openai_available() and not dry_run:
        raise ValueError(
            "langchain-openai is required for the selected LLM method(s). "
            "Run this benchmark from the project environment or install requirements.txt."
        )


def _subprocess_env(args: argparse.Namespace, domain: Optional[DomainConfig]) -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if args.clear_openai_env:
        for key in list(env):
            if key == "OPENAI_API_KEY" or key.startswith("OPENAI_"):
                env.pop(key, None)
    env["QWEN_BASE_URL"] = _normalize_qwen_base_url(args.qwen_base_url)
    if domain is not None:
        env["NOVELTY_AGENT_DB"] = str(domain.archived_db.resolve())
    return env


def _load_review_packet(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _float_equal(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) < 1e-9
    except Exception:
        return False


def review_packet_matches_step(path: Path, step: RunStep) -> bool:
    payload = _load_review_packet(path)
    if not payload:
        return False
    run = dict(payload.get("run") or {})
    config = dict(run.get("config") or {})
    future_prefilter = dict(config.get("future_prefilter") or {})
    method_names = list(run.get("method_names") or [])
    metrics = dict(run.get("metrics") or {})
    return all(
        [
            run.get("status") == "completed",
            int(metrics.get("n_task_evaluations") or 0) > 0,
            run.get("snapshot_id") == step.domain.snapshot_id,
            method_names == [step.method],
            config.get("seeds") == PROTOCOL["seeds"],
            config.get("hypotheses_per_target") == PROTOCOL["hypotheses_per_target"],
            config.get("n_gold_future_papers") == PROTOCOL["n_gold_future_papers"],
            run.get("cutoff_date") == PROTOCOL["cutoff_date"],
            run.get("future_window_start") == PROTOCOL["future_window_start"],
            run.get("future_window_end") == PROTOCOL["future_window_end"],
            bool(config.get("disable_leakage_check")) is True,
            config.get("cue_source_snapshot_id") == step.domain.snapshot_id,
            future_prefilter.get("semantic_query") == step.domain.future_semantic_query,
            _float_equal(future_prefilter.get("semantic_threshold"), PROTOCOL["future_semantic_threshold"]),
        ]
    )


def completed_review_packet(step: RunStep) -> Optional[Path]:
    if not step.output_dir.exists():
        return None
    for packet in sorted(step.output_dir.glob("*_review_packet.json")):
        if review_packet_matches_step(packet, step):
            return packet
    return None


def benchmark_cache_path_for_step(step: RunStep, args: argparse.Namespace) -> Optional[Path]:
    if getattr(args, "disable_benchmark_cache", False):
        return None
    return step.output_dir.parent / "_benchmark_cache" / "benchmark.json"


def build_retrospective_command(step: RunStep, args: argparse.Namespace) -> List[str]:
    backend_url = f"http://127.0.0.1:{args.backend_port}"
    command = [
        sys.executable,
        "-m",
        "novelty_app.evaluation.run_retrospective",
        "--backend-url",
        backend_url,
        "--qwen-base-url",
        args.qwen_base_url,
        "--data-json",
        str(ARCHIVE_DATA_JSON),
        "--data-dir",
        str(ARCHIVE_DATA_DIR),
        "--existing-snapshot-id",
        step.domain.snapshot_id,
        "--cutoff-date",
        PROTOCOL["cutoff_date"],
        "--future-window-start",
        PROTOCOL["future_window_start"],
        "--future-window-end",
        PROTOCOL["future_window_end"],
        "--n-gap-targets",
        str(PROTOCOL["n_gap_targets"]),
        "--n-cluster-pair-targets",
        str(PROTOCOL["n_cluster_pair_targets"]),
        "--n-gold-future-papers",
        str(PROTOCOL["n_gold_future_papers"]),
        "--methods",
        step.method,
        "--seeds",
        str(PROTOCOL["seeds"]),
        "--hypotheses-per-target",
        str(PROTOCOL["hypotheses_per_target"]),
        "--output-dir",
        str(step.output_dir),
        "--discovery-cue-text",
        step.domain.discovery_cue_text,
        "--cue-source-snapshot-id",
        step.domain.snapshot_id,
        "--future-semantic-query",
        step.domain.future_semantic_query,
        "--future-semantic-threshold",
        str(PROTOCOL["future_semantic_threshold"]),
        "--disable-leakage-check",
    ]
    cache_path = benchmark_cache_path_for_step(step, args)
    if cache_path is not None:
        command.extend(["--benchmark-cache-path", str(cache_path)])
    return command


def _backend_health(port: int, timeout_s: float = 2.0) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout_s) as response:
            if int(getattr(response, "status", 200)) >= 400:
                return None
            data = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    try:
        return dict(json.loads(data))
    except Exception:
        return None


def _health_db_matches(health: Dict[str, Any], db_path: Path) -> bool:
    reported = str(health.get("db_path") or "").strip()
    if not reported:
        return False
    try:
        return Path(reported).resolve() == db_path.resolve()
    except Exception:
        return False


def _health_qwen_matches(health: Dict[str, Any], qwen_base_url: str) -> bool:
    reported = _normalize_qwen_base_url(health.get("qwen_base_url"))
    expected = _normalize_qwen_base_url(qwen_base_url)
    return bool(reported) and reported == expected


class ManagedBackend:
    def __init__(self, *, domain: DomainConfig, port: int, env: Dict[str, str], log_path: Path) -> None:
        self.domain = domain
        self.port = port
        self.env = env
        self.log_path = log_path
        self.process: Optional[subprocess.Popen[Any]] = None
        self._log_handle: Optional[Any] = None
        self.reused_existing = False

    def __enter__(self) -> "ManagedBackend":
        health = _backend_health(self.port)
        expected_qwen_base_url = str(self.env.get("QWEN_BASE_URL") or "")
        if health:
            if not _health_db_matches(health, self.domain.archived_db):
                raise RuntimeError(
                    f"Backend port {self.port} is already in use with a different DB: "
                    f"{health.get('db_path')}"
                )
            if "qwen_base_url" not in health:
                raise RuntimeError(
                    f"Backend port {self.port} is already in use for {self.domain.name}, but it does not "
                    "report QWEN_BASE_URL. Stop that backend or rerun with a different --backend-port."
                )
            if not _health_qwen_matches(health, expected_qwen_base_url):
                raise RuntimeError(
                    f"Backend port {self.port} is already in use with a different QWEN_BASE_URL: "
                    f"{health.get('qwen_base_url')} (expected {expected_qwen_base_url})"
                )
            self.reused_existing = True
            return self

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        self._log_handle.write(f"\n[{_utc_now_iso()}] Starting backend for {self.domain.name}\n")
        self._log_handle.write(f"[{_utc_now_iso()}] QWEN_BASE_URL={expected_qwen_base_url}\n")
        self._log_handle.flush()
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            "agents.backend_api:app",
            "--app-dir",
            "novelty_app",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=self.env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
        )
        self._wait_until_ready()
        return self

    def _wait_until_ready(self, timeout_s: float = 90.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"Backend for {self.domain.name} exited early with code {self.process.returncode}. "
                    f"See {self.log_path}."
                )
            health = _backend_health(self.port)
            if (
                health
                and _health_db_matches(health, self.domain.archived_db)
                and _health_qwen_matches(health, str(self.env.get("QWEN_BASE_URL") or ""))
            ):
                return
            time.sleep(0.5)
        raise TimeoutError(f"Backend for {self.domain.name} did not become ready. See {self.log_path}.")

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        if self._log_handle is not None:
            self._log_handle.write(f"[{_utc_now_iso()}] Backend stopped for {self.domain.name}\n")
            self._log_handle.close()


def _manifest_payload(step: RunStep, args: argparse.Namespace, status: str, **extra: Any) -> Dict[str, Any]:
    return {
        "status": status,
        "updated_at": _utc_now_iso(),
        "domain": asdict(step.domain),
        "method": step.method,
        "output_dir": str(step.output_dir),
        "qwen_base_url": args.qwen_base_url,
        "backend_port": args.backend_port,
        "benchmark_cache_path": str(benchmark_cache_path_for_step(step, args) or ""),
        "protocol": dict(PROTOCOL),
        **extra,
    }


def _run_step(step: RunStep, args: argparse.Namespace, env: Dict[str, str]) -> int:
    step.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = step.output_dir / "step_manifest.json"
    log_path = step.output_dir / "step.log"
    command = build_retrospective_command(step, args)
    _write_json(manifest_path, _manifest_payload(step, args, "running", command=command, started_at=_utc_now_iso()))
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now_iso()}] Running {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    status = "completed" if result.returncode == 0 else "failed"
    _write_json(
        manifest_path,
        _manifest_payload(
            step,
            args,
            status,
            command=command,
            completed_at=_utc_now_iso(),
            returncode=result.returncode,
            log_path=str(log_path),
        ),
    )
    return int(result.returncode)


def _group_steps_by_domain(steps: Sequence[RunStep]) -> Dict[str, List[RunStep]]:
    grouped: Dict[str, List[RunStep]] = {}
    for step in steps:
        grouped.setdefault(step.domain.name, []).append(step)
    return grouped


def run_all(args: argparse.Namespace) -> int:
    validate_runtime_inputs(args, dry_run=bool(args.dry_run))
    steps = build_steps(args)
    if args.dry_run:
        print(f"Dry run: {len(steps)} step(s) planned.")
        for step in steps:
            print(f"- {step.domain.name} / {step.method} -> {step.output_dir}")
            print("  " + " ".join(build_retrospective_command(step, args)))
        if _selected_methods_require_openai([step.method for step in steps]):
            env = _subprocess_env(args, domain=None)
            if not env.get("OPENAI_API_KEY"):
                print("Warning: selected methods require OPENAI_API_KEY for real execution.", file=sys.stderr)
        return 0

    grouped = _group_steps_by_domain(steps)
    for domain_name, domain_steps in grouped.items():
        domain = DOMAIN_CONFIGS[domain_name]
        pending_steps: List[RunStep] = []
        for step in domain_steps:
            packet = None if args.force else completed_review_packet(step)
            if packet:
                print(f"[skip] {domain.name}/{step.method}: matching review packet exists at {packet}")
                step.output_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    step.output_dir / "step_manifest.json",
                    _manifest_payload(step, args, "skipped", matched_review_packet=str(packet)),
                )
            else:
                pending_steps.append(step)
        if not pending_steps:
            continue

        env = _subprocess_env(args, domain=domain)
        domain_output_dir = pending_steps[0].output_dir.parent
        backend_log = domain_output_dir / "backend.log"
        with ManagedBackend(domain=domain, port=args.backend_port, env=env, log_path=backend_log):
            for step in pending_steps:
                print(f"[run] {domain.name}/{step.method}")
                returncode = _run_step(step, args, env)
                if returncode != 0:
                    print(f"[failed] {domain.name}/{step.method}; see {step.output_dir / 'step.log'}", file=sys.stderr)
                    return returncode
                print(f"[done] {domain.name}/{step.method}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        return run_all(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

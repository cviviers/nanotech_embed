"""Run with python -m novelty_app.evaluation.run_controlled_study --help."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .controlled_study import (
    CONDITIONS, VERSION, BudgetExceeded, ModelCaller, StudyConfig,
    assess, choose, digest, generate, read_json, verify_task, write_json,
)


def load_manifest(root):
    value = read_json(root / "manifest.json")
    if value["version"] != VERSION or value["manifest_hash"] != digest(
            {k: v for k, v in value.items() if k != "manifest_hash"}):
        raise ValueError("Study manifest version/hash mismatch")
    for task in value["tasks"]:
        verify_task(task)
    return value


def stage_path(root, stage, task_id, condition, replicate):
    return root / stage / f"{task_id}_{condition}_{replicate}.json"


def generation_rows(root, manifest, config):
    for task in manifest["tasks"]:
        for condition in CONDITIONS:
            for replicate in range(config.replicates):
                path = stage_path(root, "generation", task["task_id"], condition, replicate)
                if not path.exists():
                    raise ValueError(f"Generation stage incomplete: {path}")
                row = read_json(path)
                if row["task_hash"] != task["task_hash"] or row["config"] != asdict(config):
                    raise ValueError("Generation provenance mismatch")
                yield task, condition, replicate, row


def prepare(args):
    from .controlled_sources import IneligibleTask, domain_sources, historical_pairs, prepare_task
    if (args.output_dir / "manifest.json").exists():
        return {"status": "already_prepared", "tasks": len(load_manifest(args.output_dir)["tasks"])}
    config = StudyConfig(replicates=args.replicates, generation_model=args.generation_model,
                         assessment_model=args.assessment_model, reasoning_effort=args.reasoning_effort)
    config.validate()
    tasks, excluded, inventory = [], [], []
    for domain, store, manifest in domain_sources(args.source_root, args.qwen_url):
        pairs = historical_pairs(store, manifest["snapshot_id"])
        inventory.append({"domain": domain, "snapshot_id": manifest["snapshot_id"], "pairs": len(pairs)})
        if args.execute and not args.dry_run:
            for a, b, gaps in pairs:
                try:
                    tasks.append(prepare_task(domain, store, manifest, a, b, gaps))
                except IneligibleTask as exc:
                    excluded.append({"domain": domain, "cluster_a": a, "cluster_b": b, "reason": str(exc)})
                # Service/schema failures abort rather than silently excluding scientific tasks.
    if not args.execute or args.dry_run:
        return {"dry_run": True, "inventory": inventory, "eligibility": "checked during pack preparation",
                "maximum_hypotheses": sum(r["pairs"] for r in inventory) * 3 * config.replicates * 3}
    if not tasks:
        raise ValueError("No eligible historical tasks")
    result = {"version": VERSION, "config": asdict(config), "tasks": tasks, "excluded": excluded,
              "inventory": inventory, "source_root": str(args.source_root.resolve()),
              "future_window_start": "2020-01-01", "future_window_end": "2026-01-01"}
    result["manifest_hash"] = digest(result)
    write_json(args.output_dir / "manifest.json", result)
    return {"tasks": len(tasks), "excluded": len(excluded), "inventory": inventory}


def generate_stage(root, manifest, config, caller):
    for task in manifest["tasks"]:
        for condition in CONDITIONS:
            for replicate in range(config.replicates):
                path = stage_path(root, "generation", task["task_id"], condition, replicate)
                if path.exists():
                    saved = read_json(path)
                    if saved["task_hash"] != task["task_hash"] or saved["config"] != asdict(config):
                        raise ValueError("Existing generation belongs to another task/configuration")
                    continue
                row = {"task_id": task["task_id"], "task_hash": task["task_hash"], "domain": task["domain"],
                       "condition": condition, "replicate": replicate, "config": asdict(config), "candidates": []}
                try:
                    row["candidates"] = generate(task, condition, replicate, caller)
                    row["missing_slots"] = config.candidates - len(row["candidates"])
                except BudgetExceeded:
                    raise
                except RuntimeError as exc:
                    row["error"] = str(exc)
                    row["missing_slots"] = config.candidates
                write_json(path, row)


def assess_stage(root, manifest, config, caller):
    for task, condition, replicate, generated in generation_rows(root, manifest, config):
        path = stage_path(root, "assessment", task["task_id"], condition, replicate)
        if path.exists():
            if read_json(path)["generation_hash"] != digest(generated):
                raise ValueError("Existing assessment belongs to another generation")
            continue
        scores, errors = {}, {}
        for candidate in generated["candidates"]:
            try:
                scores[candidate["candidate_id"]] = assess(task, candidate, caller)
            except BudgetExceeded:
                raise
            except (RuntimeError, ValueError) as exc:
                errors[candidate["candidate_id"]] = str(exc)
        write_json(path, {"task_hash": task["task_hash"], "generation_hash": digest(generated),
                          "scores": scores, "errors": errors,
                          "selected": choose(generated["candidates"], scores)})
    # This lock makes subsequent editing of generation/assessment observable to matching.
    hashes = {str(p.relative_to(root)): digest(read_json(p))
              for stage in ("generation", "assessment") for p in sorted((root / stage).glob("*.json"))}
    write_json(root / "assessment_lock.json", {"manifest_hash": manifest["manifest_hash"], "files": hashes})


def validate_lock(root, manifest):
    lock = read_json(root / "assessment_lock.json")
    if lock["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("Assessment lock belongs to another manifest")
    actual = {str(p.relative_to(root)): digest(read_json(p))
              for stage in ("generation", "assessment") for p in sorted((root / stage).glob("*.json"))}
    if lock["files"] != actual:
        raise ValueError("Generation or source assessment changed after locking")


def match_stage(args, manifest, config, caller):
    from .controlled_matching import matching_sample, assess_publications
    import hashlib
    from .candidate_match import build_corpus_index
    from .controlled_sources import retrieve_later
    from .qwen_client import QwenClient
    from .time_split import load_dataset_and_embeddings, split_corpus_by_time
    root = args.output_dir
    validate_lock(root, manifest)
    sample = matching_sample(manifest)
    # Lock corpus files, not only filenames, before any matching calls.
    inputs = {}
    for path in (args.data_json, args.data_dir / "qwen_embeddings.npy"):
        with path.open("rb") as stream:
            hasher = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        inputs[str(path.resolve())] = hasher.hexdigest()
    matching_config = {"inputs": inputs, "window": [manifest["future_window_start"], manifest["future_window_end"]],
                       "lexical": 60, "semantic": 120, "fusion_k": 60, "rerank": 64, "assess": 10,
                       "qwen_url": args.qwen_url, "version": VERSION,
                       "protocol": "sampled-selected-joint-v1", "sample_task_ids": sample,
                       "sample_limit": 100, "selection": "source_assessor_selected",
                       "manifest_hash": manifest["manifest_hash"]}
    lock_path = root / "matching_config.json"
    if lock_path.exists() and read_json(lock_path) != matching_config:
        raise ValueError("Matching inputs/settings changed; create a new experiment version")
    write_json(lock_path, matching_config)
    df, embeddings = load_dataset_and_embeddings(args.data_json, args.data_dir, ["qwen"])
    for column in ("publication_month", "publication_day"):
        if column not in df:
            df[column] = float("nan")
    split = split_corpus_by_time(df, embeddings, cutoff_date="2019-12-31",
                                future_window_start=manifest["future_window_start"],
                                future_window_end=manifest["future_window_end"])
    corpus = build_corpus_index(split.future.df.fillna(""), split.future.embeddings["qwen"])
    qwen = QwenClient(args.qwen_url)
    for task, condition, replicate, row in generation_rows(root, manifest, config):
        redundancy_path = stage_path(root, "redundancy", task["task_id"], condition, replicate)
        if not redundancy_path.exists():
            import numpy as np
            texts = [c["title"] + "\n" + c["text"] for c in row["candidates"]]
            mean_similarity = None
            if len(texts) >= 2:
                vectors = np.asarray(qwen.embed(texts, normalize=True), dtype=float)
                if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
                    raise ValueError("Invalid redundancy embeddings")
                vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
                similarities = vectors @ vectors.T
                mean_similarity = float(similarities[np.triu_indices(len(vectors), 1)].mean())
            write_json(redundancy_path, {"mean_pairwise_cosine": mean_similarity,
                                        "generation_hash": digest(row)})
        if task["task_id"] not in sample:
            continue
        selected = read_json(stage_path(root, "assessment", task["task_id"], condition, replicate))["selected"]
        depth = 10
        for candidate in row["candidates"]:
            if candidate["candidate_id"] != selected:
                continue
            path = root / "matches" / (candidate["candidate_id"] + ".json")
            if path.exists():
                if read_json(path).get("matching_config_hash") != digest(matching_config):
                    raise ValueError("Existing match belongs to another protocol")
                continue
            retrieval_path = root / "retrieval" / (candidate["candidate_id"] + ".json")
            if retrieval_path.exists():
                retrieved = read_json(retrieval_path)
            else:
                retrieved = retrieve_later(candidate, corpus, qwen)
                write_json(retrieval_path, retrieved)
            outcomes, errors = [], []
            try:
                outcomes = assess_publications(candidate, retrieved[:depth], caller,
                                               [candidate["candidate_id"], digest(matching_config), "joint"])
            except BudgetExceeded:
                raise
            except (RuntimeError, ValueError) as exc:
                errors.append({"error": str(exc)})
            write_json(path, {"candidate_id": candidate["candidate_id"], "depth": depth,
                              "matching_config_hash": digest(matching_config),
                              "retrieved_count": len(retrieved[:depth]),
                              "outcomes": outcomes, "errors": errors})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "generate", "assess", "match", "analyse", "calibrate", "diagnose", "legacy"])
    parser.add_argument("--source-root", type=Path, default=Path("data/nanomedicine"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/nanomedicine/controlled_study_runs/v1"))
    parser.add_argument("--replicates", type=int, default=3, help="Used only when preparing a new manifest")
    parser.add_argument("--generation-model", default="gpt-5.6-luna", help="Preparation only; frozen thereafter")
    parser.add_argument("--assessment-model", default="gpt-5.6-luna", help="Preparation only; frozen thereafter")
    parser.add_argument("--reasoning-effort", default="medium", help="Preparation only; frozen thereafter")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-calls", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--qwen-url", default="http://localhost:8000")
    parser.add_argument("--data-json", type=Path, default=Path("data/nanomedicine/data/cleaned_dataset.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/nanomedicine/data"))
    parser.add_argument("--ratings", type=Path, help="Matched original reader-study records, documented in experiment README")
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        print(prepare(args))
        return
    if args.stage == "legacy":
        from .controlled_reports import legacy_report
        if args.execute and not args.dry_run:
            legacy_report(args.source_root, args.output_dir)
        else:
            print("Read-only legacy analysis; --execute writes separate reports only")
        return
    manifest = load_manifest(args.output_dir)
    config = StudyConfig(**manifest["config"])
    config.validate()
    if not args.execute or args.dry_run:
        from .controlled_matching import matching_sample
        print({"stage": args.stage, "tasks": len(manifest["tasks"]), "config": asdict(config),
               "hypotheses": len(manifest["tasks"]) * 3 * config.replicates * 3,
               "generation_calls": len(manifest["tasks"]) * config.replicates * 5,
               "assessment_calls": len(manifest["tasks"]) * config.replicates * 9,
               "matching_tasks": len(matching_sample(manifest)),
               "maximum_matching_calls": len(matching_sample(manifest)) * len(CONDITIONS) * config.replicates,
               "note": "No API calls. Budgets reserve UTF-8 input bytes + output cap per attempt; retries up to 3."})
        return
    # Reject concurrent writers; stages are intentionally sequential and resumable.
    lock = args.output_dir / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        import os
        stream.write(str(os.getpid()))
    try:
        if args.stage == "analyse":
            from .controlled_reports import analyse
            analyse(args.output_dir, manifest, config)
            return
        caller = ModelCaller(args.output_dir, config, args.max_calls, args.max_tokens)
        if args.stage == "generate":
            generate_stage(args.output_dir, manifest, config, caller)
        elif args.stage == "assess":
            assess_stage(args.output_dir, manifest, config, caller)
        elif args.stage == "match":
            match_stage(args, manifest, config, caller)
        else:
            from .controlled_reports import calibrate, diagnose
            if args.stage == "calibrate":
                if not args.ratings:
                    raise ValueError("--ratings is required; existing ratings will not be modified")
                calibrate(args.ratings, args.output_dir, caller)
            else:
                diagnose(args.output_dir, manifest, config, caller)
        print({"calls_this_invocation": caller.calls, "reserved_tokens": caller.tokens})
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

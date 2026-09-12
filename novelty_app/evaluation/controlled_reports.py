"""Analysis and diagnostics; legacy observations are always read-only."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from .controlled_study import DIMENSIONS, CONDITIONS, assess, digest, read_json, supported, write_json


def paired_bootstrap(domain_values: dict[str, list[float]], samples=10000, seed=42):
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(v, dtype=float) for v in domain_values.values() if v]
    if not arrays:
        return {"difference": None, "ci95": None, "tasks": 0}
    boot = np.zeros(samples)
    for values in arrays:
        # Bound memory even for large historical target inventories.
        for start in range(0, samples, 250):
            stop = min(start + 250, samples)
            boot[start:stop] += rng.choice(values, size=(stop - start, len(values)), replace=True).mean(axis=1)
    boot /= len(arrays)
    return {"difference": float(np.mean([v.mean() for v in arrays])),
            "ci95": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
            "tasks": sum(map(len, arrays)), "domains": len(arrays)}


def analyse(root, manifest, config):
    from .controlled_matching import matching_sample
    from .run_controlled_study import generation_rows, stage_path, validate_lock
    validate_lock(root, manifest)
    sample = set(matching_sample(manifest))
    matching_config = read_json(root / "matching_config.json") if (root / "matching_config.json").exists() else None
    if matching_config and (matching_config.get("protocol") != "sampled-selected-joint-v1" or
                            set(matching_config.get("sample_task_ids", [])) != sample or
                            matching_config.get("manifest_hash") != manifest["manifest_hash"]):
        raise ValueError("Analysis requires the sampled selected-candidate matching protocol")
    rows = []
    candidate_rows = []
    for task, condition, replicate, generation in generation_rows(root, manifest, config):
        assessed = read_json(stage_path(root, "assessment", task["task_id"], condition, replicate))
        if assessed["generation_hash"] != digest(generation):
            raise ValueError("Assessment/generation mismatch")
        scores = assessed["scores"]
        success = sum(supported(s) for s in scores.values())
        row = {"task_id": task["task_id"], "domain": task["domain"], "condition": condition,
               "replicate": replicate, "generated": len(generation["candidates"]), "assessed": len(scores),
               "supported": success, "supported_yield": success / config.candidates,
               "supported_among_assessed": success / len(scores) if scores else None,
               "any_supported": int(success > 0), "selected": assessed["selected"],
               "selection_coverage": int(assessed["selected"] is not None)}
        row["matching_sample"] = task["task_id"] in sample
        redundancy = stage_path(root, "redundancy", task["task_id"], condition, replicate)
        row["mean_pairwise_cosine"] = read_json(redundancy)["mean_pairwise_cosine"] if redundancy.exists() else None
        for d in DIMENSIONS:
            values = [s[d]["score"] for s in scores.values() if s[d]["score"] is not None]
            row[d] = float(np.mean(values)) if values else None
            row[d + "_insufficient"] = sum(s[d]["insufficient_evidence"] for s in scores.values())
        rows.append(row)
        sources = {p["paper_id"]: p for p in task["papers"]}
        for candidate in generation["candidates"]:
            ids = candidate["support_citations"]
            cited = {sources[pid]["cluster_id"] for pid in ids if pid in sources}
            c = {"task_id": task["task_id"], "domain": task["domain"], "condition": condition,
                 "replicate": replicate, "candidate_id": candidate["candidate_id"],
                 "first": candidate["position"] == 0, "selected": candidate["candidate_id"] == assessed["selected"],
                 "both_communities": all(task["target"][k] in cited for k in ("cluster_a", "cluster_b")),
                 "citations": len(ids), "invalid_citations": sum(pid not in sources for pid in ids)}
            c["matching_eligible"] = task["task_id"] in sample and c["selected"]
            matching = root / "matches" / (candidate["candidate_id"] + ".json")
            if matching.exists():
                result = read_json(matching)
                if not c["matching_eligible"] or not matching_config or result.get("matching_config_hash") != digest(matching_config):
                    raise ValueError("Match output outside the frozen secondary analysis")
                matches = [o for o in result["outcomes"] if o["rank"] <= 10]
                c["match_errors"] = sum(e.get("rank", 1) <= 10 for e in result["errors"])
                for label in ("shared_proposal", "related_topic"):
                    c[label] = any(o["correspondence"] == label for o in matches)
                for label in ("supporting", "contradictory", "mixed", "insufficient_information"):
                    c[label] = any(o["outcome"] == label for o in matches)
            candidate_rows.append(c)
    task_means = {}
    for row in rows:
        task_means.setdefault((row["domain"], row["task_id"], row["condition"]), []).append(row)
    estimates = {}
    for comparator in ("direct", "summary"):
        for endpoint in ("supported_among_assessed", "supported_yield", "any_supported"):
            differences = defaultdict(list)
            for task in manifest["tasks"]:
                def mean(condition):
                    values = [r[endpoint] for r in task_means[(task["domain"], task["task_id"], condition)]
                              if r[endpoint] is not None]
                    return float(np.mean(values)) if values else None
                left, right = mean("contrastive"), mean(comparator)
                if left is not None and right is not None:
                    differences[task["domain"]].append(left - right)
            estimates[f"contrastive_minus_{comparator}:{endpoint}"] = paired_bootstrap(
                differences, config.bootstrap_samples, config.analysis_seed)
    summary = {}
    for condition in CONDITIONS:
        group = [r for r in rows if r["condition"] == condition]
        domain_means = []
        for domain in sorted({r["domain"] for r in group}):
            means = []
            for (d, _, c), values in task_means.items():
                valid = [r["supported_among_assessed"] for r in values if r["supported_among_assessed"] is not None]
                if d == domain and c == condition and valid:
                    means.append(float(np.mean(valid)))
            if means:
                domain_means.append(float(np.mean(means)))
        summary[condition] = {"generated": sum(r["generated"] for r in group),
                              "assessed": sum(r["assessed"] for r in group),
                              "supported_rate": float(np.mean(domain_means)) if domain_means else None,
                              "requested": len(group) * config.candidates,
                              "selected_sets": sum(r["selection_coverage"] for r in group)}
    correspondence_summary = {}
    for condition in CONDITIONS:
        for policy in ("selected",):
            eligible = [r for r in candidate_rows if r["condition"] == condition and r["matching_eligible"]]
            complete = [r for r in eligible if "shared_proposal" in r and r["match_errors"] == 0]
            record = {"eligible_candidates": len(eligible), "completely_matched": len(complete)}
            sampled_sets = [r for r in rows if r["condition"] == condition and r["matching_sample"]]
            record.update(sampled_tasks=len(sample), requested_sets=len(sampled_sets),
                          abstained_sets=sum(r["selected"] is None for r in sampled_sets),
                          missing_or_failed_matches=len(eligible) - len(complete))
            for endpoint in ("shared_proposal", "supporting", "contradictory", "mixed"):
                task_groups = defaultdict(list)
                for r in complete:
                    task_groups[(r["domain"], r["task_id"], r["replicate"])].append(float(r[endpoint]))
                task_values = defaultdict(list)
                for (domain, tid, rep), values in task_groups.items():
                    task_values[(domain, tid)].append(float(np.mean(values)))
                domain_values = defaultdict(list)
                for (domain, _), values in task_values.items():
                    domain_values[domain].append(float(np.mean(values)))
                record[endpoint] = float(np.mean([np.mean(v) for v in domain_values.values()])) if domain_values else None
            correspondence_summary[f"{condition}:{policy}"] = record
    usage = []
    for path in sorted((root / "calls").glob("*.json")):
        call = read_json(path)
        usage.extend({"role": call["request"]["role"], **a} for a in call["attempts"])
    report = {"summary": summary, "paired_differences": estimates, "correspondence_by_selection_policy": correspondence_summary, "replicate_rows": rows,
              "candidate_rows": candidate_rows, "call_attempts": usage,
              "matching_sample_task_ids": sorted(sample),
              "interpretation": "Model-assessed synthesis, not independent human validation. Later correspondence concerns only source-selected hypotheses in a fixed domain-balanced task sample; it is retrieval-conditioned, not forecasting. Unassessed candidates are not failures; correspondence across all candidates is not estimated."}
    write_json(root / "reports" / "analysis.json", report)
    import csv
    for name, records in (("replicates", rows), ("candidates", candidate_rows)):
        fields = sorted({k for r in records for k in r})
        with (root / "reports" / (name + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    lines = ["# Controlled study results", "", report["interpretation"], "",
             "| Condition | Generated / requested | Assessed | Supported synthesis |", "|---|---:|---:|---:|"]
    for condition, s in summary.items():
        rate = "unavailable" if s["supported_rate"] is None else f"{s['supported_rate']:.1%}"
        lines.append(f"| {condition} | {s['generated']}/{s['requested']} | {s['assessed']} | {rate} |")
    lines += ["", "Rates average replicates within task, tasks within domain, then domains equally.",
              "Paired intervals and complete dimension/failure/selection counts are in analysis.json and CSVs.",
              "Source-sharing between targets can induce dependence beyond the target bootstrap."]
    (root / "reports" / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(CONDITIONS, [summary[c]["supported_rate"] or 0 for c in CONDITIONS])
    ax.set(ylabel="Model-assessed supported synthesis", ylim=(0, 1), title="Fixed-evidence comparison")
    fig.tight_layout()
    fig.savefig(root / "reports" / "synthesis_rates.pdf")
    plt.close(fig)


def diagnose(root, manifest, config, caller):
    from .run_controlled_study import generation_rows
    from .controlled_study import ASSESSMENT_SYSTEM, SynthesisAssessment, freeze_papers
    # Frozen synthetic controls: no biomedical validity is assumed for the fictional materials.
    papers = freeze_papers([
        {"paper_id": "a", "title": "Material A", "abstract": "In buffer, material A released dye only below pH 6. No tissue experiment was performed.", "cluster_id": 1},
        {"paper_id": "b", "title": "Material B", "abstract": "Material B retained encapsulated beads for 24 hours in buffer. Combining it with material A was not tested.", "cluster_id": 2}])
    task = {"task_id": "synthetic", "cue": "Combine two materials", "target": {"cluster_a": 1, "cluster_b": 2},
            "papers": papers, "pack_hash": digest(papers)}
    task["task_hash"] = digest(task)
    fixtures = [
        ("positive", "A releases dye below pH 6; B retains beads. Test whether embedding A in B retains pH-dependent release by comparing pH 5 and 7 against A alone. Compatibility is an untested assumption.", ["a", "b"]),
        ("unsupported", "A and B have been proven to cure infections in mice through immune memory.", ["a", "b"]),
        ("invalid_ids", "A and B cure infection.", ["missing"]),
        ("non_testable", "A and B might be useful together somehow.", ["a", "b"]),
        ("contradiction", "A releases dye only above pH 8, and B was clinically tested.", ["a", "b"]),
    ]
    outputs = []
    for name, text, citations in fixtures:
        candidate = {"candidate_id": name, "title": name, "text": text, "support_citations": citations}
        score = assess(task, candidate, caller, "synthetic-v1")
        outputs.append({"fixture": name, "scores": score, "supported": supported(score)})
    write_json(root / "reports" / "synthetic_diagnostics.json", outputs)
    if not (root / "generation").exists():
        return  # Run synthetic diagnostics before any scientific hypotheses are generated.
    perturbations = []
    for task, condition, replicate, generation in generation_rows(root, manifest, config):
        for c in generation["candidates"]:
            if int(digest(c["candidate_id"])[:8], 16) % 10:
                continue
            for variant in ("repeat", "remove_a", "remove_b"):
                changed = deepcopy(task)
                if variant != "repeat":
                    cluster = changed["target"]["cluster_a" if variant == "remove_a" else "cluster_b"]
                    changed["papers"] = [p for p in changed["papers"] if p["cluster_id"] != cluster]
                changed["pack_hash"] = digest(changed["papers"])
                changed["task_hash"] = digest({k: v for k, v in changed.items() if k != "task_hash"})
                # Removed citations remain visible in the hypothesis; source support must not be invented.
                result = assess(changed, c, caller, variant)
                perturbations.append({"candidate_id": c["candidate_id"], "variant": variant, "scores": result})
    write_json(root / "reports" / "perturbations.json", perturbations)


def calibrate(ratings_path, root, caller):
    from .judge import HypothesisIdeaScoresOut, SYSTEM_IDEA_SCORER, IDEA_SCORE_FIELDS
    from scipy.stats import pearsonr, spearmanr
    records = read_json(ratings_path)
    outputs = []
    seen = set()
    for record in records:
        idea_id = record["idea_id"]
        if idea_id in seen:
            raise ValueError("Calibration input must contain one row per original idea")
        seen.add(idea_id)
        payload = {"hypotheses": [{"hypothesis_id": idea_id, **{k: record["hypothesis"][k]
                                  for k in ("title", "text", "support_citations")}}],
                   "evidence_pack": record["evidence_pack"], "cue": record.get("cue", {})}
        scores = caller.call("assessment", ["legacy-calibration", idea_id, digest(payload)],
                             SYSTEM_IDEA_SCORER, payload, HypothesisIdeaScoresOut)
        matched = [s for s in scores["scored_hypotheses"] if s["hypothesis_id"] == idea_id]
        if len(matched) != 1:
            raise ValueError("Calibration hypothesis ID mismatch")
        outputs.append({"idea_id": idea_id, "human_scores": record["human_scores"], "luna_scores": matched[0]})
    summary = {}
    for criterion in IDEA_SCORE_FIELDS:
        pairs = [(r["human_scores"].get(criterion), r["luna_scores"][criterion]["score"]) for r in outputs]
        pairs = [(float(h), float(m)) for h, m in pairs if h is not None]
        h, m = np.array(pairs).T if pairs else (np.array([]), np.array([]))
        valid = len(h) >= 3 and np.ptp(h) > 0 and np.ptp(m) > 0
        summary[criterion] = {"n": len(pairs), "human_mean": float(h.mean()) if len(h) else None,
                              "model_mean": float(m.mean()) if len(m) else None,
                              "model_minus_human": float((m-h).mean()) if len(h) else None,
                              "pearson": float(pearsonr(h, m)[0]) if valid else None,
                              "spearman": float(spearmanr(h, m)[0]) if valid else None}
    write_json(root / "reports" / "reader_calibration.json", {"input_hash": digest(records), "summary": summary, "rows": outputs})


def stored_match_sensitivity(rows):
    """Perturb settings on retained matches, without reselecting candidates/tasks."""
    from .judge import judge_candidate_match
    defaults = {"strong_rank": .80, "strong_overlap": .45, "strong_combined": .70,
                "partial_rank": .58, "partial_overlap": .22, "partial_combined": .50,
                "background_rank": .38, "background_overlap": .15}
    variants = {f"{key}{delta:+.2f}": {"thresholds": {key: value + delta}}
                for key, value in defaults.items() for delta in (-.05, .05)}
    for index, name in enumerate(("rank", "overlap", "embedding")):
        for factor in (.8, 1.2):
            weights = [.55, .25, .20]
            weights[index] *= factor
            variants[f"{name}_weight_x{factor}"] = {"weights": tuple(weights)}
    counts = defaultdict(lambda: {"available": 0, "missing": 0, "baseline_disagreements": 0,
                                  "label_changes": {name: 0 for name in variants}})
    for row in rows:
        for field in ("historical_match", "future_match"):
            group = counts[f"{row['method_name']}:{field}"]
            candidate = row.get(field)
            fingerprint = row["hypothesis"].get("idea_fingerprint")
            if not candidate or not fingerprint or any(k not in candidate for k in ("reranker_score", "embedding_score", "judge")):
                group["missing"] += 1
                continue
            baseline = judge_candidate_match(fingerprint, candidate)["label"]
            group["available"] += 1
            group["baseline_disagreements"] += baseline != candidate["judge"].get("label")
            for name, settings in variants.items():
                group["label_changes"][name] += judge_candidate_match(fingerprint, candidate, **settings)["label"] != baseline
    return {"scope": "Conditional on archived oracle-selected tasks and retained matches. No retrieval, candidate reselection or recovery-rate re-estimation.",
            "variants": variants, "groups": dict(counts)}


def legacy_report(source_root: Path, root: Path):
    """Regenerate descriptive legacy metrics without rewriting archived bundles."""
    from .metrics import select_best_task_rows, aggregate_match_metrics
    from scipy.stats import fisher_exact
    domains = defaultdict(list)
    files = sorted((source_root / "baseline_runs").rglob("*_assessment_bundle_v1.json"))
    if not files:
        raise ValueError("No legacy baseline assessment bundles found")
    for path in files:
        domain = path.relative_to(source_root / "baseline_runs").parts[0]
        bundle = read_json(path)
        for idea in bundle["ideas"]:
            for evaluation in idea["benchmark_context"]["evaluations"]:
                domains[domain].append({**idea["run_context"], **evaluation, "domain": domain,
                                       "idea_scores": idea["judge_context"]["idea_scores"],
                                       "discovery_cue": idea["discovery_cue"],
                                       "hypothesis": idea["hypothesis"], "target": idea["target"],
                                       "papers": idea["ideation_context"]["evidence_papers"]})
    selected = {domain: select_best_task_rows(rows) for domain, rows in domains.items()}
    pooled = [r for rows in selected.values() for r in rows]
    write_json(root / "reports" / "legacy_stored_match_sensitivity.json", stored_match_sensitivity(pooled))
    # Existing metrics keys omit domain. Namespace IDs on a reporting copy only so
    # the same later paper in two domain tasks is not accidentally collapsed.
    metrics = aggregate_match_metrics([{**r, "gold_future_paper_id": f"{r['domain']}:{r.get('gold_future_paper_id')}"}
                                       for r in pooled])
    table = [[0, 0], [0, 0]]
    for row in pooled:
        target = row["target"].get("effective_target", {})
        if row["method_name"] != "orchestrator" or target.get("target_type") != "cluster_pair":
            continue
        cited = set(row["hypothesis"]["support_citations"])
        clusters = {p.get("cluster_id") for p in row["papers"] if p["paper_id"] in cited}
        both = target.get("cluster_a") in clusters and target.get("cluster_b") in clusters
        table[0 if both else 1][0 if row.get("recovery_label") == "gold_recovered" else 1] += 1
    differences = {}
    for comparator in ("single_shot_llm", "retrieval_summary_direct", "cue_retrieval_generation", "random_target_control"):
        per_domain = defaultdict(list)
        for domain, rows in selected.items():
            left = {(r["seed"], r.get("gold_future_paper_id")): r for r in rows if r["method_name"] == "orchestrator"}
            right = {(r["seed"], r.get("gold_future_paper_id")): r for r in rows if r["method_name"] == comparator}
            per_target = defaultdict(list)
            for key in left.keys() & right.keys():
                a, b = left[key], right[key]
                per_target[a["target_id"]].append(int(a.get("recovery_label") == "gold_recovered") - int(b.get("recovery_label") == "gold_recovered"))
            per_domain[domain] = [float(np.mean(v)) for v in per_target.values()]
        differences[comparator] = paired_bootstrap(per_domain)
    write_json(root / "reports" / "legacy_reanalysis.json", {"metrics": metrics, "paired_target_differences": differences,
               "bridge_table": table, "fisher_p": float(fisher_exact(table)[1]),
               "note": "Historical oracle selection preserved. Source bundles never modified. Matching sensitivity requires full candidate components, not aggregate scores."})
    methods = sorted(metrics["by_method"])
    lines = ["# Legacy benchmark reanalysis", "", "Oracle-selected results; archived outputs unchanged.", "",
             "| Method | Successful tasks | Strict recovery | R@10 | MRR |", "|---|---:|---:|---:|---:|"]
    for method in methods:
        m = metrics["by_method"][method]
        lines.append(f"| {method} | {m['n_task_evaluations']} | {m['gold_recovered_rate']:.1%} | {m['gold_recall_at_10']:.1%} | {m['gold_mrr']:.3f} |")
    lines += ["", f"Exploratory both-cluster association: {table}; Fisher p={fisher_exact(table)[1]:.6f}.",
              "Paired uncertainty uses target means within domain; this estimand differs from pooled task rates.",
              "", "`legacy_stored_match_sensitivity.json` reports label changes under threshold ±0.05 and normalized weight ±20% perturbations. These are conditional on retained matches, not rerun recovery rates."]
    (root / "reports" / "LEGACY_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = []
    for domain, rows in selected.items():
        dm = aggregate_match_metrics(rows)["by_method"]
        matrix.append([dm[m]["gold_recovered_rate"] if m in dm else np.nan for m in methods])
    fig, ax = plt.subplots(figsize=(12, 5))
    display = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
    ax.set_yticks(range(len(selected)), list(selected))
    for i, values in enumerate(matrix):
        for j, value in enumerate(values):
            ax.text(j, i, f"{value:.1%}" if np.isfinite(value) else "missing", ha="center", va="center")
    fig.colorbar(display, ax=ax, label="Strict recovery")
    fig.tight_layout()
    fig.savefig(root / "reports" / "legacy_all_seven_methods.pdf")
    plt.close(fig)

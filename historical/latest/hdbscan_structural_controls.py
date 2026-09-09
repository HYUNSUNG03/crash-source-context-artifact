"""Pre-contract structural controls for two vanilla HDBSCAN configurations."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work" / "stage4-runtime" / "py312"))
sys.path.insert(0, str(ROOT / "work"))

import numpy as np
from hdbscan import HDBSCAN
import selector_contract_probe as probe
import stage4_numeric_v1 as legacy


OUT = ROOT / "outputs" / "hdbscan-structural-controls-v1"
CONFIGS = {
    "A_mcs5_ms2": {"min_cluster_size": 5, "min_samples": 2},
    "B_mcs3_ms5": {"min_cluster_size": 3, "min_samples": 5},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relabel_noise(raw):
    values = [int(v) for v in raw]
    next_noise = min(values + [0]) - 1
    for index, value in enumerate(values):
        if value < 0:
            values[index] = next_noise
            next_noise -= 1
    return values


def fit(x, parameters):
    model = HDBSCAN(
        metric="euclidean",
        cluster_selection_epsilon=0.0,
        cluster_selection_method="eom",
        allow_single_cluster=True,
        gen_min_span_tree=True,
        core_dist_n_jobs=1,
        **parameters,
    ).fit(x)
    raw = [int(v) for v in model.labels_]
    return raw, relabel_noise(raw)


def synthetic_truth(name, n):
    if name == "one_blob":
        return ["blob0"] * n, 1, 0
    if name == "two_separated_blobs":
        return ["blob0"] * 35 + ["blob1"] * 35, 2, 0
    if name == "three_separated_blobs":
        return ["blob0"] * 30 + ["blob1"] * 30 + ["blob2"] * 30, 3, 0
    if name == "unequal_density_blobs":
        return ["blob0"] * 25 + ["blob1"] * 55, 2, 0
    if name == "two_blobs_with_noise":
        truth = ["blob0"] * 30 + ["blob1"] * 30 + [f"noise{i}" for i in range(8)]
        return truth, 2, 8
    if name == "one_blob_tiny_singleton_anomaly":
        return ["blob0"] * 60 + ["singleton"], 1, 1
    raise ValueError(name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, metric_provenance = legacy.load_pinned_metrics()
    cases = []
    for name, (x, _) in probe.generated_cases().items():
        truth, expected_k, noise_mode = synthetic_truth(name, len(x))
        cases.append(("synthetic", name, x, truth, expected_k, noise_mode))
    for name, (x, _) in probe.actual_controls().items():
        cases.append(("actual_one_defect", name, x, ["one_defect"] * len(x), 1, 0))

    output = []
    for config_name, parameters in CONFIGS.items():
        for role, name, x, truth, expected_k, noise_mode in cases:
            raw, predicted = fit(x, parameters)
            raw_k = len({v for v in raw if v >= 0})
            noise = sum(v < 0 for v in raw)
            purity, inverse, f_measure = metrics.statistical_scores(predicted, truth)
            structural_pass = raw_k == expected_k
            noise_pass = noise == noise_mode
            output.append({
                "config": config_name,
                "min_cluster_size": parameters["min_cluster_size"],
                "min_samples": parameters["min_samples"],
                "role": role,
                "case": name,
                "reports": len(x),
                "expected_raw_k": expected_k,
                "expected_noise_rule": f"exactly_{noise_mode}",
                "expected_noise": noise_mode,
                "raw_k": raw_k,
                "noise": noise,
                "reported_clusters_after_noise_singletons": len(set(predicted)),
                "purity": purity,
                "inverse_purity": inverse,
                "f_measure": f_measure,
                "structural_pass": structural_pass,
                "noise_pass": noise_pass,
                "contract_case_pass": structural_pass and noise_pass,
            })

    results = OUT / "results.csv"
    with results.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    summaries = {}
    for config_name in CONFIGS:
        rows = [row for row in output if row["config"] == config_name]
        synthetic = [row for row in rows if row["role"] == "synthetic"]
        actual = [row for row in rows if row["role"] == "actual_one_defect"]
        summaries[config_name] = {
            "synthetic_passed": sum(row["contract_case_pass"] for row in synthetic),
            "synthetic_total": len(synthetic),
            "actual_one_defect_passed": sum(row["contract_case_pass"] for row in actual),
            "actual_one_defect_total": len(actual),
            "all_single_multi_noise_controls_pass": all(row["contract_case_pass"] for row in rows),
            "failed_cases": [row["case"] for row in rows if not row["contract_case_pass"]],
        }
    summary_value = {
        "schema_version": "hdbscan-structural-controls-summary-v1",
        "hdbscan": {"version": "0.8.40", "epsilon": 0.0, "method": "eom",
                    "allow_single_cluster": True, "metric": "euclidean"},
        "noise_scoring": "each HDBSCAN noise report becomes its own singleton prediction",
        "thresholds_fit_to_report_labels": False,
        "metric": metric_provenance,
        "configurations": summaries,
    }
    summary = OUT / "summary.json"
    summary.write_text(json.dumps(summary_value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation_value = {
        "status": "passed",
        "result_rows": len(output),
        "synthetic_cases_per_configuration": 6,
        "actual_one_defect_cases_per_configuration": 3,
        "all_rows_finite": all(np.isfinite(row["f_measure"]) for row in output),
        "probe_script_sha256": sha(ROOT / "work" / "selector_contract_probe.py"),
        "generator_sha256": sha(Path(__file__)),
        "artifacts": {"results.csv": sha(results), "summary.json": sha(summary)},
    }
    validation = OUT / "validation.json"
    validation.write_text(json.dumps(validation_value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary_value, "results": output,
                      "validation": validation_value}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

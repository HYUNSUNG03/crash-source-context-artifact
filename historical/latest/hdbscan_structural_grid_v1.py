"""Label-blind structural HDBSCAN grid on frozen full-152 Stage-4 matrices.

Phase 1 opens only frozen label-free inputs/vectors and freezes every
epsilon=0 EOM partition.  Phase 2 verifies that freeze before opening the
legacy ground-truth labels solely for GPTrace metric scoring.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work" / "stage4-runtime" / "py312"))
INPUT = ROOT / "outputs" / "full152-stage4-inputs-v2"
SOURCE = ROOT / "outputs" / "full152-stage4-results-v2"
OUTPUT = ROOT / "outputs" / "hdbscan-structural-grid-v1"
LABELS = Path(r"/RESEARCH_WORKSPACES\2026-09-05\new-chat-2\outputs\revised\contexts-v4.jsonl")
PRIMARY = ("L_f1_W0", "L_f1_W5", "L_f3_W0", "L_f3_W5")
REPRESENTATIONS = ("B", "BL", "BD4", "BLD4")
GRID = tuple((mcs, ms) for mcs in (2, 3, 4, 5, 8) for ms in (1, 2, 3, 5))
DIMENSION = 384


def load_legacy():
    path = ROOT / "work" / "stage4_numeric_v1.py"
    spec = importlib.util.spec_from_file_location("structural_grid_legacy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


L = load_legacy()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(L.canonical_json(row) + "\n" for row in rows), encoding="utf-8", newline="\n")


def load_source_label_free():
    freeze = load_json(SOURCE / "phase1-freeze.json")
    if freeze.get("labels_loaded") is not False or freeze.get("status") != "frozen_label_blind":
        raise ValueError("source_phase1_not_label_blind")
    for name, digest in freeze["artifacts"].items():
        if sha(SOURCE / name) != digest:
            raise ValueError(f"source_artifact_hash_mismatch:{name}")
    rows = load_jsonl(INPUT / "encoding-inputs.jsonl")
    matrices = load_jsonl(SOURCE / "matrices.jsonl")
    import numpy as np
    with np.load(SOURCE / "bge-vectors.npz", allow_pickle=False) as archive:
        hashes = [str(value) for value in archive["hashes"].tolist()]
        values = archive["vectors"]
    if values.shape != (len(hashes), DIMENSION):
        raise ValueError("source_vector_shape_mismatch")
    return freeze, rows, matrices, {key: value for key, value in zip(hashes, values)}


def fit(matrix, min_cluster_size: int, min_samples: int) -> dict:
    import numpy as np
    from hdbscan import HDBSCAN

    model = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean",
        cluster_selection_epsilon=0.0, gen_min_span_tree=True, allow_single_cluster=True,
        cluster_selection_method="eom", core_dist_n_jobs=4,
    ).fit(matrix)
    raw = [int(value) for value in model.labels_]
    partition = L.relabel_noise(raw)
    try:
        dbcv = float(model.relative_validity_)
    except (AttributeError, ValueError):
        dbcv = -1.0
    persistence_values = [float(value) for value in model.cluster_persistence_]
    persistence = float(np.mean(persistence_values)) if persistence_values else 0.0
    if not math.isfinite(dbcv):
        dbcv = -1.0
    if not math.isfinite(persistence):
        persistence = 0.0
    return {
        "epsilon": 0.0, "dbcv": dbcv, "persistence": persistence,
        "cluster_persistence": persistence_values,
        "reported_cluster_count": len(set(partition)),
        "raw_non_noise_structural_cluster_count": len({value for value in raw if value >= 0}),
        "noise_point_count": sum(value < 0 for value in raw),
        "raw_partition_sha256": L.partition_hash(raw), "partition_sha256": L.partition_hash(partition),
        "raw_partition": raw, "partition": partition,
    }


def phase1() -> None:
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    freeze_path = OUTPUT / "phase1-freeze.json"
    if freeze_path.exists():
        raise ValueError("phase1_already_exists")
    source_freeze, rows, matrices, vectors = load_source_label_free()
    if importlib.metadata.version("hdbscan") != "0.8.40":
        raise ValueError("hdbscan_version_mismatch")
    import numpy as np
    fused = L.fuse_rows(rows, vectors)
    results: list[dict] = []
    computed: dict[tuple[str, int, int], dict] = {}
    for matrix_record in matrices:
        condition, target = matrix_record["condition"], matrix_record["target"]
        representation, report_ids, key = matrix_record["representation"], matrix_record["report_ids"], matrix_record["key"]
        matrix = np.stack([fused[(condition, rid)]["values"][representation] for rid in report_ids]).astype(np.float64, copy=False)
        matrix_hash = L.sha256_bytes(np.ascontiguousarray(matrix).tobytes())
        if matrix_hash != matrix_record["matrix_sha256"]:
            raise ValueError(f"matrix_rebuild_hash_mismatch:{key}")
        for min_cluster_size, min_samples in GRID:
            cache_key = (matrix_hash, min_cluster_size, min_samples)
            if cache_key in computed:
                candidate = computed[cache_key]
                reused_from = candidate["key"]
            else:
                # This grid runs one fixed epsilon=0 fit per structure pair;
                # use index 0 only to retain the frozen selection schema.
                candidate = {"key": key, "evaluation_index": 0, **fit(matrix, min_cluster_size, min_samples)}
                computed[cache_key] = candidate
                reused_from = None
            annotation = L.candidate_annotation(candidate)
            results.append({
                "schema_version": "hdbscan-structural-grid-selection-v1", "key": key,
                "condition": condition, "condition_tier": matrix_record["condition_tier"],
                "target": target, "representation": representation, "reports": len(report_ids),
                "report_ids": report_ids, "report_ids_sha256": matrix_record["report_ids_sha256"],
                "matrix_sha256": matrix_hash, "partition": candidate["partition"], **annotation,
                "min_cluster_size": min_cluster_size, "min_samples": min_samples,
                "cluster_selection_method": "eom", "cluster_selection_epsilon": 0.0,
                "allow_single_cluster": True, "reused_from": reused_from, "labels_loaded": False,
            })
        print(f"Frozen {key}: 20 structural parameter pairs", flush=True)
    expected = len(matrices) * len(GRID)
    if len(results) != expected:
        raise ValueError(f"unexpected_selection_count:{len(results)}:{expected}")
    selections_path = OUTPUT / "selections.jsonl"
    write_jsonl(selections_path, results)
    manifest = {
        "schema_version": "hdbscan-structural-grid-phase1-freeze-v1", "status": "frozen_label_blind",
        "labels_loaded": False, "purpose": "label-blind epsilon-zero EOM structural-parameter grid",
        "source_phase1_sha256": sha(SOURCE / "phase1-freeze.json"),
        "source_artifacts": {name: sha(SOURCE / name) for name in ("bge-vectors.npz", "matrices.jsonl")},
        "input_sha256": sha(INPUT / "encoding-inputs.jsonl"),
        "hdbscan": {"version": "0.8.40", "metric": "euclidean", "cluster_selection_method": "eom",
                    "cluster_selection_epsilon": 0.0, "allow_single_cluster": True,
                    "min_cluster_size_values": [2, 3, 4, 5, 8], "min_samples_values": [1, 2, 3, 5]},
        "matrix_count": len(matrices), "unique_matrix_count": source_freeze["unique_matrix_count"],
        "parameter_pair_count": len(GRID), "selection_count": len(results), "computed_fit_count": len(computed),
        "selections_sha256": sha(selections_path),
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(freeze_path, manifest)
    print(json.dumps({"phase": "phase1_complete", "matrices": len(matrices), "parameter_pairs": len(GRID),
                      "fits": len(computed), "selections": len(results), "labels_loaded": False}, indent=2))


def verify_phase1():
    manifest = load_json(OUTPUT / "phase1-freeze.json")
    if manifest.get("status") != "frozen_label_blind" or manifest.get("labels_loaded") is not False:
        raise ValueError("bad_phase1_manifest")
    if sha(Path(manifest["generator"]["path"])) != manifest["generator"]["sha256"]:
        raise ValueError("generator_changed")
    if sha(SOURCE / "phase1-freeze.json") != manifest["source_phase1_sha256"]:
        raise ValueError("source_phase1_changed")
    for name, digest in manifest["source_artifacts"].items():
        if sha(SOURCE / name) != digest:
            raise ValueError(f"source_changed:{name}")
    if sha(INPUT / "encoding-inputs.jsonl") != manifest["input_sha256"]:
        raise ValueError("input_changed")
    if sha(OUTPUT / "selections.jsonl") != manifest["selections_sha256"]:
        raise ValueError("selections_changed")
    selections = load_jsonl(OUTPUT / "selections.jsonl")
    if len(selections) != manifest["selection_count"]:
        raise ValueError("selection_count_mismatch")
    expected_pairs = set(GRID)
    for row in selections:
        if row["labels_loaded"] is not False or row["epsilon"] != 0.0 or row["cluster_selection_epsilon"] != 0.0:
            raise ValueError(f"invalid_selection:{row['key']}")
        if (row["min_cluster_size"], row["min_samples"]) not in expected_pairs:
            raise ValueError(f"unexpected_parameters:{row['key']}")
        if L.partition_hash(row["partition"]) != row["partition_sha256"]:
            raise ValueError(f"partition_hash_mismatch:{row['key']}")
    return manifest, selections


def score_one(metrics, matrix, predicted, truth):
    return L.score_metrics(metrics, matrix, predicted, truth)


def phase2() -> None:
    started = time.monotonic()
    manifest, selections = verify_phase1()
    print("Verified structural-grid Phase 1; opening labels for scoring", flush=True)
    label_rows = load_jsonl(LABELS)
    labels = {row["report_id"]: row for row in label_rows}
    frozen_ids = {rid for row in selections for rid in row["report_ids"]}
    if set(labels) != frozen_ids:
        raise ValueError("label_inventory_mismatch")
    _, input_rows, _, vectors = load_source_label_free()
    fused = L.fuse_rows(input_rows, vectors)
    metrics, metric_provenance = L.load_pinned_metrics()
    import numpy as np
    score_rows = []
    for row in selections:
        truth = [labels[rid]["label"] for rid in row["report_ids"]]
        matrix = np.stack([fused[(row["condition"], rid)]["values"][row["representation"]] for rid in row["report_ids"]]).astype(np.float64, copy=False)
        if L.sha256_bytes(np.ascontiguousarray(matrix).tobytes()) != row["matrix_sha256"]:
            raise ValueError(f"phase2_matrix_hash_mismatch:{row['key']}")
        predicted = [int(value) for value in row["partition"]]
        score_rows.append({
            "condition": row["condition"], "condition_tier": row["condition_tier"], "target": row["target"],
            "representation": row["representation"], "reports": row["reports"],
            "min_cluster_size": row["min_cluster_size"], "min_samples": row["min_samples"],
            "true_bugs": len(set(truth)), "predicted_clusters": len(set(predicted)),
            "raw_structural_clusters": row["raw_structural_clusters"], "noise_points": row["noise_points"],
            "matrix_sha256": row["matrix_sha256"], "partition_sha256": row["partition_sha256"],
            "epsilon": 0.0, "relative_validity_hdbscan": row["relative_validity_hdbscan"],
            "persistence": row["persistence"], "single_cluster_correction": row["single_cluster_correction"],
            **score_one(metrics, matrix, predicted, truth),
        })
    scores_path = OUTPUT / "scores.csv"
    with scores_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0])); writer.writeheader(); writer.writerows(score_rows)
    summaries, tradeoffs = [], []
    for mcs, ms in GRID:
        scoped = [row for row in score_rows if row["min_cluster_size"] == mcs and row["min_samples"] == ms]
        tradeoffs.append({
            "min_cluster_size": mcs, "min_samples": ms, "matrices": len(scoped),
            "macro_f_measure": float(np.mean([row["f_measure"] for row in scoped])),
            "macro_purity": float(np.mean([row["purity"] for row in scoped])),
            "macro_inverse_purity": float(np.mean([row["inverse_purity"] for row in scoped])),
            "mean_predicted_clusters": float(np.mean([row["predicted_clusters"] for row in scoped])),
            "mean_true_bugs": float(np.mean([row["true_bugs"] for row in scoped])),
            "overcluster_matrices": sum(row["predicted_clusters"] > row["true_bugs"] for row in scoped),
            "undercluster_matrices": sum(row["predicted_clusters"] < row["true_bugs"] for row in scoped),
            "exact_cluster_count_matrices": sum(row["predicted_clusters"] == row["true_bugs"] for row in scoped),
            "noise_matrices": sum(row["noise_points"] > 0 for row in scoped),
            "noise_points": sum(row["noise_points"] for row in scoped),
            "single_cluster_matrices": sum(row["predicted_clusters"] == 1 for row in scoped),
            "single_cluster_correction_matrices": sum(bool(row["single_cluster_correction"]) for row in scoped),
        })
        for condition in PRIMARY:
            for representation in REPRESENTATIONS:
                matching = [row for row in scoped if row["condition"] == condition and row["representation"] == representation]
                summaries.append({
                    "min_cluster_size": mcs, "min_samples": ms, "condition": condition,
                    "representation": representation, "targets": len(matching), "reports": sum(row["reports"] for row in matching),
                    "macro_f_measure": float(np.mean([row["f_measure"] for row in matching])),
                    "macro_purity": float(np.mean([row["purity"] for row in matching])),
                    "macro_inverse_purity": float(np.mean([row["inverse_purity"] for row in matching])),
                    "mean_predicted_clusters": float(np.mean([row["predicted_clusters"] for row in matching])),
                    "noise_matrices": sum(row["noise_points"] > 0 for row in matching),
                    "single_cluster_matrices": sum(row["predicted_clusters"] == 1 for row in matching),
                })
    summary_path = OUTPUT / "score-summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    tradeoff_path = OUTPUT / "tradeoffs.csv"
    with tradeoff_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tradeoffs[0])); writer.writeheader(); writer.writerows(tradeoffs)
    f3_w5_rows = [row for row in score_rows if row["condition"] == "L_f3_W5"]
    f3_path = OUTPUT / "f3-w5-per-target.csv"
    with f3_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(f3_w5_rows[0])); writer.writeheader(); writer.writerows(f3_w5_rows)
    write_json(OUTPUT / "phase2-manifest.json", {
        "schema_version": "hdbscan-structural-grid-phase2-manifest-v1", "status": "scored_after_phase1_verification",
        "phase1_sha256": sha(OUTPUT / "phase1-freeze.json"), "phase1_verified_before_label_read": True,
        "label_source": {"path": str(LABELS), "sha256": sha(LABELS)}, "gptrace_metric": metric_provenance,
        "score_rows": len(score_rows), "outputs": {path.name: sha(path) for path in (scores_path, summary_path, tradeoff_path, f3_path)},
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "elapsed_seconds": time.monotonic() - started,
    })
    print(json.dumps({"phase": "phase2_complete", "score_rows": len(score_rows), "parameter_pairs": len(GRID)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("phase1", "phase2"))
    args = parser.parse_args()
    phase1() if args.phase == "phase1" else phase2()


if __name__ == "__main__":
    main()

"""Label-blind probe for the HDBSCAN single-cluster correction contract.

This is an independent diagnostic.  It reads frozen candidate/vector artifacts,
never reads defect labels, and never writes under outputs/.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "work" / "stage4-runtime" / "py312"
sys.path.insert(0, str(RUNTIME))

import numpy as np
from hdbscan import HDBSCAN
from sklearn.metrics import pairwise_distances


OLD = Path(r"/RESEARCH_WORKSPACES\2026-09-05\new-chat-2")
OLD_REP = OLD / "outputs" / "dual-frame-pilot-v1" / "representations.npz"
OLD_INPUTS = OLD / "outputs" / "dual-frame-pilot-v1" / "inputs.jsonl"
OLD_CANDIDATES = OLD / "outputs" / "dual-frame-pilot-v1-ext" / "candidates.jsonl"
OLD_MATRICES = OLD / "outputs" / "dual-frame-pilot-v1-ext" / "matrices.jsonl"


def rows(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def partition_hash(values) -> str:
    payload = json.dumps([int(x) for x in values], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def correction(item: dict) -> bool:
    return (int(item["raw_non_noise_structural_cluster_count"]) == 1
            and int(item["noise_point_count"]) == 0
            and abs(float(item["dbcv"]) - 0.5) <= 1e-12)


def fit(x: np.ndarray, epsilon: float) -> dict:
    model = HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
        cluster_selection_epsilon=float(epsilon),
        gen_min_span_tree=True,
        allow_single_cluster=True,
        cluster_selection_method="eom",
        core_dist_n_jobs=1,
    ).fit(x)
    raw = np.asarray(model.labels_, dtype=int)
    try:
        dbcv = float(model.relative_validity_)
    except (AttributeError, ValueError):
        dbcv = -1.0
    persistence_values = [float(v) for v in model.cluster_persistence_]
    return {
        "epsilon": float(epsilon),
        "dbcv": dbcv if math.isfinite(dbcv) else -1.0,
        "persistence": float(np.mean(persistence_values)) if persistence_values else 0.0,
        "cluster_persistence": persistence_values,
        "raw_non_noise_structural_cluster_count": len(set(raw[raw >= 0])),
        "noise_point_count": int(np.sum(raw < 0)),
        "reported_cluster_count": len(set(raw)),
        "raw_partition": raw.tolist(),
        "raw_partition_sha256": partition_hash(raw),
    }


def sweep(x: np.ndarray, steps: int = 100) -> list[dict]:
    distances = pairwise_distances(x, metric="euclidean")
    nonzero = distances[distances > 0]
    if not len(nonzero):
        item = fit(x, 0.0)
        item["evaluation_index"] = 0
        return [item]
    stack = [(float(nonzero.min()), float(nonzero.max()))]
    cached = {}
    for _ in range(steps):
        if not stack:
            break
        start, end = stack.pop(0)
        ends = []
        for epsilon in (start, end):
            key = round(epsilon, 15)
            if key not in cached:
                item = fit(x, epsilon)
                item["evaluation_index"] = len(cached)
                cached[key] = item
            ends.append(cached[key])
        left, right = ends
        if (left["reported_cluster_count"] != right["reported_cluster_count"]
                or left["dbcv"] != right["dbcv"]):
            middle = (start + end) / 2
            step = (middle - start) / steps
            if start + step < middle:
                stack.append((start + step, middle))
            if middle + step < end - step:
                stack.append((middle + step, end - step))
    return list(cached.values())


def within_best(items, key, fraction):
    best = max(float(item[key]) for item in items)
    if best > 0:
        return [item for item in items if float(item[key]) / best >= 1 - fraction]
    tolerance = fraction * max(abs(best), 1e-12)
    return [item for item in items if float(item[key]) >= best - tolerance]


def choose_s0(items):
    unique = {}
    for item in sorted(items, key=lambda r: int(r["evaluation_index"])):
        signature = (float(item["dbcv"]), int(item["reported_cluster_count"]),
                     float(item["persistence"]))
        unique.setdefault(signature, item)
    stage1 = sorted(within_best(list(unique.values()), "dbcv", .2),
                    key=lambda r: (-float(r["dbcv"]), int(r["evaluation_index"])))[:10]
    stage2 = sorted(within_best(stage1, "persistence", .2),
                    key=lambda r: (-float(r["persistence"]), int(r["evaluation_index"])))[:10]
    return min(stage2, key=lambda r: (int(r["reported_cluster_count"]),
                                      int(r["evaluation_index"])))


def choose_smax(items):
    return min(items, key=lambda r: (-float(r["dbcv"]), -float(r["persistence"]),
                                    int(r["noise_point_count"]), float(r["epsilon"]),
                                    int(r["evaluation_index"])))


def same_structural_partition(reference: np.ndarray, observed: np.ndarray) -> bool:
    """Exact label-permutation-invariant agreement for reference core points.

    Noise points are not required to stay noise. Every reference clustered point
    must remain clustered, and all pairwise same/different-cluster relations must
    remain identical.
    """
    core = reference >= 0
    if np.any(observed[core] < 0):
        return False
    ref = reference[core]
    obs = observed[core]
    return bool(np.array_equal(ref[:, None] == ref[None, :],
                               obs[:, None] == obs[None, :]))


def loo_invariant(x: np.ndarray, candidate: dict) -> bool:
    full = np.asarray(candidate["raw_partition"], dtype=int)
    expected_k = int(candidate["raw_non_noise_structural_cluster_count"])
    if expected_k < 2:
        return False
    for removed in range(len(x)):
        keep = np.arange(len(x)) != removed
        observed = np.asarray(fit(x[keep], float(candidate["epsilon"]))["raw_partition"], dtype=int)
        if len(set(observed[observed >= 0])) != expected_k:
            return False
        if not same_structural_partition(full[keep], observed):
            return False
    return True


def choose_sloo(x: np.ndarray, items: list[dict]) -> tuple[dict, str, int]:
    """Correction is an unscored fallback; only invariant real-DBCV splits replace it."""
    base = choose_s0(items)
    if not correction(base):
        return base, "S0_not_correction", 0
    eligible = [item for item in items
                if int(item["raw_non_noise_structural_cluster_count"]) >= 2
                and math.isfinite(float(item["dbcv"])) and float(item["dbcv"]) > 0]
    unique = {}
    for item in sorted(eligible, key=lambda r: int(r["evaluation_index"])):
        unique.setdefault(item["raw_partition_sha256"], item)
    ordered = sorted(unique.values(), key=lambda r: (-float(r["dbcv"]),
                                                      -float(r["persistence"]),
                                                      int(r["noise_point_count"]),
                                                      float(r["epsilon"])))
    tested = 0
    for candidate in ordered:
        tested += 1
        if loo_invariant(x, candidate):
            return candidate, "real_DBCV_LOO_invariant_split", tested
    return base, "synthetic_correction_fallback", tested


def generated_cases():
    rng = np.random.default_rng(20260909)
    blob1 = rng.normal(0, 0.35, size=(60, 2))
    two = np.vstack([rng.normal((-3, 0), .35, size=(35, 2)),
                     rng.normal((3, 0), .35, size=(35, 2))])
    three = np.vstack([rng.normal((-4, 0), .4, size=(30, 2)),
                       rng.normal((0, 4), .4, size=(30, 2)),
                       rng.normal((4, 0), .4, size=(30, 2))])
    unequal = np.vstack([rng.normal((-2.0, 0), .16, size=(25, 2)),
                         rng.normal((2.0, 0), .75, size=(55, 2))])
    noisy = np.vstack([rng.normal((-3, 0), .35, size=(30, 2)),
                       rng.normal((3, 0), .35, size=(30, 2)),
                       rng.uniform((-7, -5), (7, 5), size=(8, 2))])
    singleton = np.vstack([rng.normal(0, .35, size=(60, 2)), np.array([[7.0, 7.0]])])
    return {
        "one_blob": (blob1, 1),
        "two_separated_blobs": (two, 2),
        "three_separated_blobs": (three, 3),
        "unequal_density_blobs": (unequal, 2),
        "two_blobs_with_noise": (noisy, 2),
        "one_blob_tiny_singleton_anomaly": (singleton, 1),
    }


def actual_controls():
    archive = np.load(OLD_REP, allow_pickle=False)
    all_ids = archive["report_ids"].tolist()
    conditions = archive["conditions"].tolist()
    values = archive["values"]
    index = {rid: i for i, rid in enumerate(all_ids)}
    targets = {row["report_id"]: row["target"] for row in rows(OLD_INPUTS)}
    matrices = {row["key"]: row for row in rows(OLD_MATRICES)}
    groups = collections.defaultdict(list)
    wanted = {
        "development/libxml2__libxml2_xml_read_memory_fuzzer/BP2",
        "development/libxml2__libxml2_xml_read_memory_fuzzer/BP2_F0F2",
        "development/libxml2__libxml2_xml_read_memory_fuzzer/BP2_F0W2",
    }
    for item in rows(OLD_CANDIDATES):
        if item["key"] in wanted:
            groups[item["key"]].append(item)
    result = {}
    for key in sorted(wanted):
        matrix = matrices[key]
        condition = matrix["condition"]
        cidx = conditions.index(condition)
        report_ids = matrix["report_ids"]
        assert all(targets[rid] == "libxml2__libxml2_xml_read_memory_fuzzer" for rid in report_ids)
        x = np.stack([values[index[rid], cidx] for rid in report_ids]).astype(np.float64)
        digest = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        if digest != matrix["matrix_sha256"]:
            raise ValueError(f"matrix hash mismatch for {key}: {digest}")
        result[key] = (x, groups[key])
    return result


def summary(name, x, items, expected_k=None):
    base = choose_s0(items)
    selected, reason, tested = choose_sloo(x, items)
    best_multi = [r for r in items if int(r["raw_non_noise_structural_cluster_count"]) >= 2]
    best_multi = choose_smax(best_multi) if best_multi else None
    return {
        "name": name,
        "n": len(x),
        "expected_structural_clusters": expected_k,
        "S0": {"k": int(base["raw_non_noise_structural_cluster_count"]),
               "noise": int(base["noise_point_count"]), "dbcv": float(base["dbcv"]),
               "correction": correction(base)},
        "best_real_multi": None if best_multi is None else {
            "k": int(best_multi["raw_non_noise_structural_cluster_count"]),
            "noise": int(best_multi["noise_point_count"]), "dbcv": float(best_multi["dbcv"])},
        "Sloo": {"k": int(selected["raw_non_noise_structural_cluster_count"]),
                 "noise": int(selected["noise_point_count"]), "dbcv": float(selected["dbcv"]),
                 "correction_fallback": correction(selected), "reason": reason,
                 "partitions_tested": tested},
        "passes_expected": expected_k is None or int(selected["raw_non_noise_structural_cluster_count"]) == expected_k,
    }


def main():
    result = {"contract": {
        "name": "Sloo-v0",
        "single_cluster_0.5_role": "unscored fallback only",
        "replacement_pool": "raw structural k>=2, finite positive real DBCV",
        "replacement_gate": "exact leave-one-out structural invariance at the candidate epsilon",
        "labels_loaded": False,
    }, "synthetic": [], "actual_one_defect_controls": []}
    for name, (x, expected) in generated_cases().items():
        result["synthetic"].append(summary(name, x, sweep(x), expected))
    for name, (x, items) in actual_controls().items():
        result["actual_one_defect_controls"].append(summary(name, x, items, 1))
    result["synthetic_pass"] = all(row["passes_expected"] for row in result["synthetic"])
    result["actual_controls_preserved"] = all(
        row["passes_expected"] for row in result["actual_one_defect_controls"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

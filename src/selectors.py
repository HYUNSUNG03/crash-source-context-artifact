# Extracted unchanged from the recorded experiment; replay requires only standard Python.
import collections
import math

def within_best(items: list[dict], key: str, fraction: float) -> tuple[list[dict], bool]:
    best = max(float(item[key]) for item in items)
    if best > 0:
        return [item for item in items if float(item[key]) / best >= 1 - fraction], False
    tolerance = fraction * max(abs(best), 1e-12)
    return [item for item in items if float(item[key]) >= best - tolerance], True

def choose_s0(items: list[dict]) -> tuple[dict, dict]:
    ordered = sorted(items, key=lambda item: int(item["evaluation_index"]))
    representatives: collections.OrderedDict[tuple[float, int, float], dict] = collections.OrderedDict()
    for item in ordered:
        signature = (float(item["dbcv"]), int(item["reported_cluster_count"]), float(item["persistence"]))
        representatives.setdefault(signature, item)
    stage_1_all, dbcv_fallback = within_best(list(representatives.values()), "dbcv", 0.2)
    stage_1 = sorted(stage_1_all, key=lambda item: (-float(item["dbcv"]), int(item["evaluation_index"])))[:10]
    stage_2_all, persistence_fallback = within_best(stage_1, "persistence", 0.2)
    stage_2 = sorted(stage_2_all, key=lambda item: (-float(item["persistence"]), int(item["evaluation_index"])))[:10]
    selected = min(stage_2, key=lambda item: (int(item["reported_cluster_count"]), int(item["evaluation_index"])))
    return selected, {
        "input_candidate_count": len(items), "metric_signature_representative_count": len(representatives),
        "dbcv_20pct_candidate_count": len(stage_1_all), "dbcv_top10_candidate_count": len(stage_1),
        "persistence_20pct_candidate_count": len(stage_2_all), "persistence_top10_candidate_count": len(stage_2),
        "nonpositive_dbcv_fallback": dbcv_fallback, "nonpositive_persistence_fallback": persistence_fallback,
        "tie_break": "lowest_evaluation_index_after_legacy_cluster_count_rule",
    }

def choose_smax(items: list[dict]) -> tuple[dict | None, dict]:
    eligible = [item for item in items if math.isfinite(float(item["dbcv"])) and math.isfinite(float(item["persistence"]))]
    if not eligible:
        return None, {"input_candidate_count": len(items), "finite_candidate_count": 0,
                      "unavailable_reason": "no_candidate_with_finite_dbcv_and_persistence"}
    selected = min(eligible, key=lambda item: (
        -float(item["dbcv"]), -float(item["persistence"]), int(item["noise_point_count"]),
        float(item["epsilon"]), int(item["evaluation_index"]),
    ))
    return selected, {
        "input_candidate_count": len(items), "finite_candidate_count": len(eligible), "unavailable_reason": None,
        "lexicographic_key": [-float(selected["dbcv"]), -float(selected["persistence"]),
                              int(selected["noise_point_count"]), float(selected["epsilon"]),
                              int(selected["evaluation_index"])],
    }

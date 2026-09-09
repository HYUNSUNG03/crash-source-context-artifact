"""Run the frozen 152-report Stage-4 primary L x D4 experiment.

Phase 1 reads only label-free inputs and freezes all HDBSCAN candidates and
S0/Smax selections.  Phase 2 verifies that freeze before opening contexts-v4
solely to obtain the ground-truth label column for GPTrace scoring.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs" / "full152-stage4-inputs-v2"
RESULTS = ROOT / "outputs" / "full152-stage4-results-v2"
OLD_CONTEXTS = Path(r"/RESEARCH_WORKSPACES\2026-09-05\new-chat-2\outputs\revised\contexts-v4.jsonl")
PRIMARY = ("L_f1_W0", "L_f1_W5", "L_f3_W0", "L_f3_W5")
DIMENSION = 384


def load_legacy():
    source = ROOT / "work" / "stage4_numeric_v1.py"
    spec = importlib.util.spec_from_file_location("full152_legacy_stage4", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


L = load_legacy()
L.INPUT_DIR = INPUT
L.RESULT_DIR = RESULTS
L.PRIMARY_CONDITIONS = PRIMARY
L.SECONDARY_CONDITIONS = ()
L.ALL_CONDITIONS = PRIMARY


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty_jsonl:{path}")
    return rows


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(L.canonical_json(row) + "\n" for row in rows), encoding="utf-8", newline="\n")


def verify_inputs() -> tuple[dict, list[dict], dict[str, str]]:
    protocol = read_json(INPUT / "protocol-with-d4.json")
    if protocol.get("schema_version") != "full152-stage4-protocol-v2":
        raise ValueError("unexpected_full152_protocol_schema")
    if protocol.get("status") != "label_free_inputs_frozen_waiting_for_phase1":
        raise ValueError("inputs_not_frozen_for_phase1")
    if protocol.get("labels_loaded") is not False:
        raise ValueError("label_free_input_contract_broken")
    if tuple(protocol.get("L", {}).get("primary_candidates", ())) != PRIMARY:
        raise ValueError("primary_condition_order_mismatch")
    encoder = protocol.get("encoder", {})
    for key, expected in {"model": L.MODEL_NAME, "revision": L.MODEL_REVISION,
                          "dimension": 384, "content_token_budget": 510,
                          "max_sequence_length": 512}.items():
        if encoder.get(key) != expected:
            raise ValueError(f"encoder_mismatch:{key}")
    frozen = protocol.get("frozen_inputs", {})
    for name in ("encoding_inputs", "unique_texts"):
        record = frozen.get(name, {})
        path = INPUT / record.get("path", "")
        if not path.is_file() or sha(path) != record.get("sha256"):
            raise ValueError(f"frozen_input_hash_mismatch:{name}")
    rows = read_jsonl(INPUT / "encoding-inputs.jsonl")
    if len(rows) != 608:
        raise ValueError(f"unexpected_encoding_rows:{len(rows)}")
    seen, expected_hashes = set(), set()
    for row in rows:
        key = (row.get("condition"), row.get("report_id"))
        if row.get("schema_version") != "full152-stage4-encoding-input-v1" or key[0] not in PRIMARY:
            raise ValueError(f"invalid_encoding_row:{key}")
        if key in seen:
            raise ValueError(f"duplicate_condition_report:{key}")
        seen.add(key)
        parts = row.get("component_text_sha256", {})
        if set(parts) != {"trace", "no_args_trace", "asan", "L", "D4"}:
            raise ValueError(f"invalid_components:{key}")
        expected_hashes.update(parts.values())
        if tuple(row.get("representations", ())) != ("B", "BL", "BD4", "BLD4"):
            raise ValueError(f"invalid_representations:{key}")
        if row.get("max_causal_depth") != 4 or row.get("D_depth_reoptimized") is not False:
            raise ValueError(f"D4_protocol_mismatch:{key}")
    by_condition = collections.Counter(row["condition"] for row in rows)
    if by_condition != collections.Counter({condition: 152 for condition in PRIMARY}):
        raise ValueError(f"primary_coverage_not_152:{dict(by_condition)}")
    text_rows = read_jsonl(INPUT / "unique-texts.jsonl")
    text_by_hash = {}
    for row in text_rows:
        digest, text = row.get("text_sha256"), row.get("text")
        if not isinstance(digest, str) or not isinstance(text, str) or not text:
            raise ValueError("invalid_unique_text")
        if L.sha256_bytes(text.encode("utf-8")) != digest or digest in text_by_hash:
            raise ValueError(f"invalid_unique_text_hash:{digest}")
        text_by_hash[digest] = text
    if set(text_by_hash) != expected_hashes:
        raise ValueError("text_inventory_not_exact")
    return protocol, rows, text_by_hash


def annotate(item: dict) -> dict:
    return L.candidate_annotation(item)


def phase1() -> None:
    started = time.monotonic()
    RESULTS.mkdir(parents=True, exist_ok=True)
    freeze = RESULTS / "phase1-freeze.json"
    if freeze.exists():
        raise ValueError("phase1_already_exists")
    protocol, rows, texts = verify_inputs()
    model_provenance = L.verify_model_snapshot()
    import importlib.metadata
    version = importlib.metadata.version("hdbscan")
    if version != "0.8.40":
        raise ValueError(f"hdbscan_version_mismatch:{version}")
    vectors, tokens, encoder = L.encode_texts(texts)
    vector_hashes = sorted(vectors)
    import numpy as np
    array = np.stack([vectors[digest] for digest in vector_hashes]).astype(np.float32, copy=False)
    vectors_path = RESULTS / "bge-vectors.npz"
    np.savez_compressed(vectors_path, hashes=np.array(vector_hashes), vectors=array)
    encoding_manifest = RESULTS / "encoding-manifest.json"
    write_json(encoding_manifest, {**encoder, "unique_texts": len(vector_hashes), "records": len(rows),
        "tokens_by_hash": tokens, "input_sha256": sha(INPUT / "encoding-inputs.jsonl"),
        "unique_texts_sha256": sha(INPUT / "unique-texts.jsonl"), "vectors_sha256": sha(vectors_path),
        "vector_hash_keys_sha256": L.sha256_bytes("\n".join(vector_hashes).encode()),
        "vector_array_float32_sha256": L.sha256_bytes(np.ascontiguousarray(array).tobytes()),
        "model_snapshot": model_provenance, "encoder_script_sha256": sha(Path(__file__))})

    fused = L.fuse_rows(rows, vectors)
    grouped: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for condition, report_id in fused:
        grouped[(condition, fused[(condition, report_id)]["target"])].append(report_id)
    matrices, candidates, selections = [], [], []
    computed: dict[str, tuple[list[dict], dict, str]] = {}
    cache_root = RESULTS / "hdbscan-cache"
    for condition in PRIMARY:
        targets = sorted(target for current, target in grouped if current == condition)
        for target in targets:
            report_ids = sorted(grouped[(condition, target)])
            for representation in ("B", "BL", "BD4", "BLD4"):
                matrix = np.stack([fused[(condition, rid)]["values"][representation] for rid in report_ids]).astype(np.float64, copy=False)
                matrix_hash = L.sha256_bytes(np.ascontiguousarray(matrix).tobytes())
                key = f"{condition}/{target}/{representation}"
                if matrix_hash in computed:
                    items, sweep, reused = computed[matrix_hash]
                    runtime_kind = "reused_matrix_lookup"
                else:
                    print(f"Sweeping {key}: {len(report_ids)} reports", flush=True)
                    items, sweep = L.enumerate_sweep(matrix, cache_root / matrix_hash)
                    computed[matrix_hash] = (items, sweep, key)
                    reused, runtime_kind = None, "computed_hdbscan_sweep"
                s0, s0_trace = L.choose_s0(items)
                smax, smax_trace = L.choose_smax(items)
                if smax is None:
                    raise ValueError(f"smax_unavailable:{key}")
                for item in items:
                    candidates.append({"schema_version": "full152-stage4-candidate-v1", "key": key,
                        "condition": condition, "condition_tier": "primary_full_coverage", "target": target,
                        "representation": representation, "report_ids_sha256": L.sha256_bytes("\n".join(report_ids).encode()),
                        "matrix_sha256": matrix_hash, "reused_from": reused, **item})
                for selector, selected, trace in (("S0", s0, s0_trace), ("Smax", smax, smax_trace)):
                    selections.append({"schema_version": "full152-stage4-selection-v1", "key": key, "selector": selector,
                        "condition": condition, "target": target, "representation": representation,
                        "matrix_sha256": matrix_hash, "report_ids": report_ids, "partition": selected["partition"],
                        **annotate(selected), "selection_trace": trace, "labels_loaded": False})
                matrices.append({"schema_version": "full152-stage4-matrix-v1", "key": key, "condition": condition,
                    "condition_tier": "primary_full_coverage", "target": target, "representation": representation,
                    "reports": len(report_ids), "report_ids": report_ids,
                    "report_ids_sha256": L.sha256_bytes("\n".join(report_ids).encode()), "matrix_sha256": matrix_hash,
                    "dimension": DIMENSION, "sweep": sweep, "candidate_count": len(items), "reused_from": reused,
                    "clustering_runtime_kind": runtime_kind,
                    "component_hashes": [{"report_id": rid, **fused[(condition, rid)]["components"]} for rid in report_ids]})
                print(f"Frozen {key}: {len(items)} candidates; S0={s0['evaluation_index']}; Smax={smax['evaluation_index']}", flush=True)
    for condition in PRIMARY:
        count = sum(1 for current, _ in fused if current == condition)
        if count != 152:
            raise ValueError(f"condition_not_full152:{condition}:{count}")
    paths = {"bge-vectors.npz": vectors_path, "encoding-manifest.json": encoding_manifest}
    for name, data in (("candidates.jsonl", candidates), ("matrices.jsonl", matrices), ("selections.jsonl", selections)):
        path = RESULTS / name
        write_jsonl(path, data)
        paths[name] = path
    manifest = {"schema_version": "full152-stage4-phase1-freeze-v1", "status": "frozen_label_blind",
        "labels_loaded": False, "protocol_sha256": sha(INPUT / "protocol-with-d4.json"),
        "input_files": {"protocol-with-d4.json": sha(INPUT / "protocol-with-d4.json"),
                        "encoding-inputs.jsonl": sha(INPUT / "encoding-inputs.jsonl"),
                        "unique-texts.jsonl": sha(INPUT / "unique-texts.jsonl"),
                        "model-download-manifest.json": sha(INPUT / "model-download-manifest.json")},
        "model": L.MODEL_NAME, "revision": L.MODEL_REVISION, "model_snapshot": model_provenance,
        "hdbscan_version": version, "selectors": ["S0", "Smax"], "condition_coverage": {c: 152 for c in PRIMARY},
        "matrix_count": len(matrices), "selection_count": len(selections), "candidate_count": len(candidates),
        "unique_matrix_count": len(computed), "artifacts": {name: sha(path) for name, path in paths.items()},
        "generator": {"file": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "elapsed_seconds": time.monotonic() - started}
    write_json(freeze, manifest)
    print(json.dumps({"phase": "phase1_complete", "matrices": len(matrices), "selections": len(selections),
                      "labels_loaded": False}, indent=2), flush=True)


def verify_phase1() -> tuple[dict, list[dict], list[dict], dict, list[dict]]:
    manifest = read_json(RESULTS / "phase1-freeze.json")
    if manifest.get("schema_version") != "full152-stage4-phase1-freeze-v1" or manifest.get("labels_loaded") is not False:
        raise ValueError("not_label_blind_phase1")
    for name, digest in manifest["input_files"].items():
        if sha(INPUT / name) != digest:
            raise ValueError(f"phase1_input_changed:{name}")
    for name, digest in manifest["artifacts"].items():
        if sha(RESULTS / name) != digest:
            raise ValueError(f"phase1_artifact_changed:{name}")
    generator = manifest["generator"]
    if sha(Path(generator["file"])) != generator["sha256"]:
        raise ValueError("generator_changed_after_freeze")
    matrices, candidates, selections = (read_jsonl(RESULTS / name) for name in
        ("matrices.jsonl", "candidates.jsonl", "selections.jsonl"))
    by_key, cand_key = collections.defaultdict(list), collections.defaultdict(list)
    for row in selections: by_key[row["key"]].append(row)
    for row in candidates: cand_key[row["key"]].append(row)
    if {row["key"] for row in matrices} != set(by_key) or set(by_key) != set(cand_key):
        raise ValueError("phase1_key_mismatch")
    for matrix in matrices:
        key = matrix["key"]
        report_ids = matrix["report_ids"]
        if report_ids != sorted(report_ids) or len(report_ids) != len(set(report_ids)):
            raise ValueError(f"bad_frozen_report_order:{key}")
        if matrix["report_ids_sha256"] != L.sha256_bytes("\n".join(report_ids).encode()):
            raise ValueError(f"bad_frozen_report_hash:{key}")
        for selection in by_key[key]:
            match = [row for row in cand_key[key] if row["evaluation_index"] == selection["evaluation_index"]]
            if len(match) != 1 or match[0]["partition"] != selection["partition"]:
                raise ValueError(f"selection_mismatch:{key}:{selection['selector']}")
            if L.partition_hash(selection["partition"]) != selection["partition_sha256"]:
                raise ValueError(f"partition_hash_mismatch:{key}:{selection['selector']}")
    import numpy as np
    with np.load(RESULTS / "bge-vectors.npz", allow_pickle=False) as archive:
        hashes, values = [str(x) for x in archive["hashes"].tolist()], archive["vectors"]
    if values.dtype != np.dtype("float32") or values.shape != (len(hashes), DIMENSION):
        raise ValueError("invalid_vector_archive")
    return manifest, matrices, selections, {h: v for h, v in zip(hashes, values)}, read_jsonl(INPUT / "encoding-inputs.jsonl")


def phase2() -> None:
    started = time.monotonic()
    manifest, matrices, selections, vectors, rows = verify_phase1()
    print("Verified Phase 1 freeze; opening ground-truth labels for scoring", flush=True)
    label_rows = read_jsonl(OLD_CONTEXTS)
    labels = {row["report_id"]: row for row in label_rows}
    report_ids = {row["report_id"] for row in rows}
    if len(labels) != len(label_rows) or set(labels) != report_ids:
        raise ValueError("label_inventory_mismatch")
    if any(labels[row["report_id"]]["target"] != row["target"] for row in rows):
        raise ValueError("label_target_mismatch")
    metrics, metric_provenance = L.load_pinned_metrics()
    fused = L.fuse_rows(rows, vectors)
    selection_by_key = collections.defaultdict(list)
    for selection in selections: selection_by_key[selection["key"]].append(selection)
    results = []
    import numpy as np
    for matrix in matrices:
        key, report_ids = matrix["key"], matrix["report_ids"]
        truth = [labels[rid]["label"] for rid in report_ids]
        vectors_matrix = np.stack([fused[(matrix["condition"], rid)]["values"][matrix["representation"]]
                                   for rid in report_ids]).astype(np.float64, copy=False)
        if L.sha256_bytes(np.ascontiguousarray(vectors_matrix).tobytes()) != matrix["matrix_sha256"]:
            raise ValueError(f"matrix_rebuild_hash_mismatch:{key}")
        for selected in sorted(selection_by_key[key], key=lambda row: row["selector"]):
            predicted = [int(value) for value in selected["partition"]]
            results.append({"condition": matrix["condition"], "condition_tier": matrix["condition_tier"],
                "target": matrix["target"], "representation": matrix["representation"], "selector": selected["selector"],
                "reports": len(report_ids), "true_bugs": len(set(truth)), "predicted_clusters": len(set(predicted)),
                "matrix_sha256": matrix["matrix_sha256"], "partition_sha256": selected["partition_sha256"],
                "evaluation_index": selected["evaluation_index"], "epsilon": selected["epsilon"],
                "relative_validity_hdbscan": selected["relative_validity_hdbscan"], "persistence": selected["persistence"],
                "single_cluster_correction": selected["single_cluster_correction"], **L.score_metrics(metrics, vectors_matrix, predicted, truth)})
    scores = RESULTS / "scores.csv"
    with scores.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    summaries = []
    import numpy as np
    for condition in PRIMARY:
        for representation in ("B", "BL", "BD4", "BLD4"):
            for selector in ("S0", "Smax"):
                matching = [row for row in results if (row["condition"], row["representation"], row["selector"]) == (condition, representation, selector)]
                aps = [row["pairwise_average_precision"] for row in matching if row["pairwise_average_precision"] is not None]
                summaries.append({"condition": condition, "condition_tier": "primary_full_coverage", "representation": representation,
                    "selector": selector, "targets": len(matching), "reports": sum(row["reports"] for row in matching),
                    "macro_purity": float(np.mean([row["purity"] for row in matching])),
                    "macro_inverse_purity": float(np.mean([row["inverse_purity"] for row in matching])),
                    "macro_f_measure": float(np.mean([row["f_measure"] for row in matching])),
                    "macro_adjusted_rand_index": float(np.mean([row["adjusted_rand_index"] for row in matching])),
                    "macro_pairwise_average_precision": float(np.mean(aps)) if aps else None})
    summary = RESULTS / "score-summary.json"; write_json(summary, {"primary_conditions": list(PRIMARY), "rows": summaries})
    phase2_manifest = {"schema_version": "full152-stage4-phase2-manifest-v1", "status": "scored_after_phase1_verification",
        "phase1_verified_before_label_read": True, "phase1_freeze_sha256": sha(RESULTS / "phase1-freeze.json"),
        "label_source": {"path": str(OLD_CONTEXTS), "sha256": sha(OLD_CONTEXTS)}, "label_inventory_exactly_matches_frozen_reports": True,
        "gptrace_metric": metric_provenance, "score_rows": len(results),
        "selected_single_cluster_correction_rows": sum(row["single_cluster_correction"] for row in results),
        "single_cluster_correction_used_as_final_f": False, "outputs": {scores.name: sha(scores), summary.name: sha(summary)},
        "generator": {"file": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))}, "elapsed_seconds": time.monotonic() - started}
    write_json(RESULTS / "phase2-manifest.json", phase2_manifest)
    print(json.dumps({"phase": "phase2_complete", "score_rows": len(results)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("phase1", "phase2")); args = parser.parse_args()
    if args.phase == "phase1": phase1()
    else: phase2()


if __name__ == "__main__":
    main()

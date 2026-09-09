"""Build label-free primary L x D4 inputs for the predeclared 152-report panel.

The first command constructs B and the four primary L conditions only.  It does
not accept a D4 file and therefore cannot accidentally use labels.  The second
command joins a separately frozen report-specific D4 artifact and emits the
same encoding-input schema used by the fixed-32 Stage-4 runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OLD_ROOT = Path(r"/RESEARCH_WORKSPACES\2026-09-05\new-chat-2")
OUT = ROOT / "outputs" / "full152-stage4-inputs-v2"
L3_CONTEXTS = OLD_ROOT / "outputs" / "l3-development-v1" / "contexts.jsonl"
BASE_INPUTS = OLD_ROOT / "outputs" / "frame-function-v1" / "numeric-base-inputs.jsonl"
FIXED_ENCODING = ROOT / "outputs" / "stage4-fixed-d4-l-inputs" / "encoding-inputs.jsonl"
FIXED_TEXTS = ROOT / "outputs" / "stage4-fixed-d4-l-inputs" / "unique-texts.jsonl"
FIXED_PROTOCOL = ROOT / "outputs" / "stage4-fixed-d4-l-inputs" / "protocol.json"
MODEL_MANIFEST = ROOT / "outputs" / "stage4-fixed-d4-l-inputs" / "model-download-manifest.json"

PRIMARY = ("L_f1_W0", "L_f1_W5", "L_f3_W0", "L_f3_W5")
SEPARATOR = "\n\n--- FRAME ---\n\n"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8", newline="\n")


def text_hash(text: str) -> str:
    return sha_bytes(text.encode("utf-8"))


def focus_only(frames: list[dict], count: int) -> str:
    """Use exactly the source line marked FOCUS from each mapped stack frame."""
    # The inherited frozen L3 rule is "first three unique usable locations";
    # when fewer exist it uses the actual count rather than substituting a
    # different frame or discarding the report (three SoX reports have one).
    selected = frames[:count]
    if not selected:
        raise ValueError("no_mapped_frames")
    lines: list[str] = []
    for frame in selected:
        marked = [line for line in frame["text"].splitlines() if line.startswith("FOCUS:")]
        if len(marked) != 1:
            raise ValueError(f"unexpected_focus_count:{frame.get('frame_number')}:{len(marked)}")
        lines.append(marked[0])
    return SEPARATOR.join(lines)


def inventory(texts: dict[str, str]) -> list[dict]:
    return [{"text_sha256": digest, "text": texts[digest]} for digest in sorted(texts)]


def input_provenance() -> dict:
    return {
        "l3_contexts": {"path": str(L3_CONTEXTS), "sha256": sha_file(L3_CONTEXTS)},
        "numeric_base_inputs": {"path": str(BASE_INPUTS), "sha256": sha_file(BASE_INPUTS)},
        "fixed32_encoding_inputs": {"path": str(FIXED_ENCODING), "sha256": sha_file(FIXED_ENCODING)},
        "fixed32_unique_texts": {"path": str(FIXED_TEXTS), "sha256": sha_file(FIXED_TEXTS)},
        "fixed32_protocol": {"path": str(FIXED_PROTOCOL), "sha256": sha_file(FIXED_PROTOCOL)},
        "model_download_manifest": {"path": str(MODEL_MANIFEST), "sha256": sha_file(MODEL_MANIFEST)},
    }


def build_base() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    l3_rows = read_jsonl(L3_CONTEXTS)
    numeric_rows = read_jsonl(BASE_INPUTS)
    if len(l3_rows) != 152:
        raise ValueError(f"expected_152_L3_contexts:{len(l3_rows)}")
    context_ids = {row["report_id"] for row in l3_rows}
    if len(context_ids) != 152:
        raise ValueError("duplicate_or_missing_context_id")
    targets = {row["report_id"]: row["target"] for row in l3_rows}
    numeric = {row["report_id"]: row for row in numeric_rows if row["report_id"] in context_ids}
    if set(numeric) != context_ids:
        raise ValueError(f"numeric_base_inventory_mismatch:{len(numeric)}")
    l3 = {row["report_id"]: row for row in l3_rows}

    base_rows: list[dict] = []
    condition_rows: list[dict] = []
    texts: dict[str, str] = {}
    for report_id in sorted(context_ids):
        base = numeric[report_id]
        if base["target"] != targets[report_id] or l3[report_id]["target"] != targets[report_id]:
            raise ValueError(f"target_mismatch:{report_id}")
        base_texts = base["texts"]
        base_hashes = base["text_hashes"]
        if set(base_texts) != {"trace", "no_args_trace", "asan"} or set(base_hashes) != set(base_texts):
            raise ValueError(f"unexpected_B_components:{report_id}")
        for name, value in base_texts.items():
            if text_hash(value) != base_hashes[name]:
                raise ValueError(f"base_text_hash_mismatch:{report_id}:{name}")
            texts[base_hashes[name]] = value
        condition_texts = {
            "L_f1_W0": focus_only(l3[report_id]["frames"], 1),
            "L_f1_W5": l3[report_id]["conditions"]["L0"]["text"],
            "L_f3_W0": focus_only(l3[report_id]["frames"], 3),
            "L_f3_W5": l3[report_id]["conditions"]["L3"]["text"],
        }
        base_rows.append({
            "schema_version": "full152-stage4-base-input-v1",
            "report_id": report_id,
            "target": targets[report_id],
            "component_text_sha256": dict(base_hashes),
        })
        for condition in PRIMARY:
            value = condition_texts[condition]
            digest = text_hash(value)
            texts[digest] = value
            condition_rows.append({
                "schema_version": "full152-stage4-l-input-v1",
                "condition": condition,
                "report_id": report_id,
                "target": targets[report_id],
                "L_text_sha256": digest,
            })

    fixed_text_by_hash = {row["text_sha256"]: row["text"] for row in read_jsonl(FIXED_TEXTS)}
    fixed_rows = read_jsonl(FIXED_ENCODING)
    fixed_by_key = {(row["condition"], row["report_id"]): row for row in fixed_rows if row["condition"] in PRIMARY}
    base_by_id = {row["report_id"]: row for row in base_rows}
    l_by_key = {(row["condition"], row["report_id"]): row for row in condition_rows}
    comparisons = []
    for key, fixed in sorted(fixed_by_key.items()):
        condition, report_id = key
        if report_id not in base_by_id:
            raise ValueError(f"fixed32_id_not_in_full152:{report_id}")
        components = fixed["component_text_sha256"]
        l_match = components["L"] == l_by_key[key]["L_text_sha256"]
        b_match = all(components[name] == base_by_id[report_id]["component_text_sha256"][name]
                      for name in ("trace", "no_args_trace", "asan"))
        # The fixed inventory lets this prove content equality as well as digest equality.
        if not l_match or fixed_text_by_hash[components["L"]] != texts[l_by_key[key]["L_text_sha256"]]:
            raise ValueError(f"fixed32_L_equivalence_failed:{condition}:{report_id}")
        if not b_match:
            raise ValueError(f"fixed32_B_equivalence_failed:{condition}:{report_id}")
        comparisons.append({"condition": condition, "report_id": report_id,
                            "L_hash_match": l_match, "B_hash_match": b_match})
    if not comparisons:
        raise ValueError("no_fixed32_overlap")

    base_path = OUT / "base-components.jsonl"
    l_path = OUT / "l-primary-components.jsonl"
    texts_path = OUT / "unique-texts-base.jsonl"
    write_jsonl(base_path, base_rows)
    write_jsonl(l_path, condition_rows)
    write_jsonl(texts_path, inventory(texts))
    shutil.copyfile(MODEL_MANIFEST, OUT / "model-download-manifest.json")
    protocol = {
        "schema_version": "full152-stage4-base-protocol-v2",
        "status": "label_free_B_and_primary_L_frozen_waiting_for_D4",
        "labels_loaded": False,
        "sample": {"reports": 152, "targets": {target: sum(row["target"] == target for row in base_rows)
                   for target in sorted(set(targets.values()))}},
        "L": {"primary_candidates": list(PRIMARY), "definitions": {
            "L_f1_W0": "first source-mapped frame, focus line only",
            "L_f1_W5": "first source-mapped frame, plus/minus 5 physical lines",
            "L_f3_W0": "first up-to-3 unique usable source-mapped frames, focus lines only",
            "L_f3_W5": "first up-to-3 unique usable source-mapped frames, each plus/minus 5 physical lines",
        }, "missing_policy": "No replacement and no fallback to another frame or range."},
        "B": "exact trace, no_args_trace, and asan texts from the old numeric-base artifact for the predeclared 152-ID subset",
        "D": {"max_causal_depth": 4, "status": "not joined in this artifact"},
        "clustering": read_json(FIXED_PROTOCOL)["clustering"],
        "encoder": read_json(FIXED_PROTOCOL)["encoder"],
        "fusion_after_d4_join": {
            "B": "u(mean(u(trace),u(no_args_trace),u(asan)))",
            "BL": "u(B+u(L))", "BD4": "u(B+u(D4))", "BLD4": "u(B+u(L)+u(D4))",
            "weights": "all present components weight 1; no tuning",
        },
        "phase_order": ["label-free B/L freeze", "separately frozen D4 join", "pinned BGE encoding",
                        "candidate/selector freeze", "label join", "GPTrace scoring"],
        "inputs": input_provenance(),
        "frozen_outputs": {},
    }
    write_json(OUT / "protocol.json", protocol)
    protocol["frozen_outputs"] = {
        "base_components": {"path": base_path.name, "sha256": sha_file(base_path)},
        "l_primary_components": {"path": l_path.name, "sha256": sha_file(l_path)},
        "unique_texts_base": {"path": texts_path.name, "sha256": sha_file(texts_path)},
    }
    write_json(OUT / "protocol.json", protocol)
    validation = {
        "schema_version": "full152-stage4-base-validation-v2", "status": "pass", "labels_loaded": False,
        "reports": len(base_rows), "L_rows": len(condition_rows), "unique_texts": len(texts),
        "target_counts": protocol["sample"]["targets"],
        "fixed32_overlap_rows": len(comparisons), "fixed32_L_hash_matches": sum(row["L_hash_match"] for row in comparisons),
        "fixed32_B_hash_matches": sum(row["B_hash_match"] for row in comparisons),
        "outputs": protocol["frozen_outputs"],
    }
    write_json(OUT / "validation.json", validation)
    print(json.dumps({"status": "base_complete", "reports": len(base_rows), "L_rows": len(condition_rows),
                      "fixed32_overlap": len(comparisons)}, indent=2))


def join_d4(path: Path) -> None:
    protocol = read_json(OUT / "protocol.json")
    if protocol.get("status") != "label_free_B_and_primary_L_frozen_waiting_for_D4":
        raise ValueError("unexpected_base_protocol_status")
    base_rows = read_jsonl(OUT / "base-components.jsonl")
    l_rows = read_jsonl(OUT / "l-primary-components.jsonl")
    texts = {row["text_sha256"]: row["text"] for row in read_jsonl(OUT / "unique-texts-base.jsonl")}
    d4_rows = read_jsonl(path)
    base_by_id = {row["report_id"]: row for row in base_rows}
    d4_by_id = {row["report_id"]: row for row in d4_rows}
    if len(base_by_id) != len(base_rows) or len(d4_by_id) != len(d4_rows):
        raise ValueError("duplicate_report_id")
    if set(d4_by_id) != set(base_by_id):
        raise ValueError(f"D4_inventory_mismatch:{len(d4_by_id)}:{len(base_by_id)}")
    finalized: list[dict] = []
    l_by_key = {(row["condition"], row["report_id"]): row for row in l_rows}
    endpoint_counts: dict[str, int] = {}
    for report_id, base in sorted(base_by_id.items()):
        d4 = d4_by_id[report_id]
        # v2 D4 is deliberately target-free: its request identity is the
        # report ID, while target comes from the independently frozen L/B
        # inventory.  Accept an omitted target, but verify it if supplied.
        if d4.get("target") not in (None, base["target"]) or not isinstance(d4.get("d4_text"), str):
            raise ValueError(f"invalid_D4_row:{report_id}")
        digest = text_hash(d4["d4_text"])
        if d4.get("d4_text_sha256") != digest:
            raise ValueError(f"D4_hash_mismatch:{report_id}")
        if d4.get("max_causal_depth") != 4:
            raise ValueError(f"D4_depth_not_4:{report_id}")
        texts[digest] = d4["d4_text"]
        endpoint = str(d4.get("endpoint_status", "unknown"))
        endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
        for condition in PRIMARY:
            finalized.append({
                "schema_version": "full152-stage4-encoding-input-v1", "condition": condition,
                "report_id": report_id, "target": base["target"], "max_causal_depth": 4,
                "D_depth_reoptimized": False, "endpoint_status": endpoint,
                "component_text_sha256": {**base["component_text_sha256"],
                    "L": l_by_key[(condition, report_id)]["L_text_sha256"], "D4": digest},
                "representations": ["B", "BL", "BD4", "BLD4"],
            })
    final_path = OUT / "encoding-inputs.jsonl"
    texts_path = OUT / "unique-texts.jsonl"
    write_jsonl(final_path, finalized)
    write_jsonl(texts_path, inventory(texts))
    final_protocol = dict(protocol)
    final_protocol.update({
        "schema_version": "full152-stage4-protocol-v2", "status": "label_free_inputs_frozen_waiting_for_phase1",
        "labels_loaded": False,
        "D": {"max_causal_depth": 4, "reoptimized": False,
              "input_path": str(path), "input_sha256": sha_file(path)},
        "frozen_inputs": {"encoding_inputs": {"path": final_path.name, "sha256": sha_file(final_path)},
                          "unique_texts": {"path": texts_path.name, "sha256": sha_file(texts_path)}},
    })
    write_json(OUT / "protocol-with-d4.json", final_protocol)
    validation = {
        "schema_version": "full152-stage4-input-validation-v2", "status": "pass", "labels_loaded": False, "reports": len(base_rows),
        "encoding_rows": len(finalized), "unique_texts": len(texts), "endpoint_status_counts": endpoint_counts,
        "D4_input": {"path": str(path), "sha256": sha_file(path)},
        "outputs": {"encoding_inputs": sha_file(final_path),
                                               "unique_texts": sha_file(texts_path),
                                               "protocol": sha_file(OUT / "protocol-with-d4.json")},
    }
    write_json(OUT / "validation-with-d4.json", validation)
    print(json.dumps({"status": "D4_join_complete", "reports": len(base_rows),
                      "encoding_rows": len(finalized), "endpoint_status_counts": endpoint_counts}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--build-base", action="store_true")
    actions.add_argument("--join-d4", type=Path)
    args = parser.parse_args()
    if args.build_base:
        build_base()
    else:
        join_d4(args.join_d4)


if __name__ == "__main__":
    main()

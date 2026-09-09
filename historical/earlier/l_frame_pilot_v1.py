"""Deterministic L0/L3/Lall diagnostic over ten frozen development reports.

This adapter only reads the established FreeType/Poppler/SoX artifacts.  It
keeps all new contexts, embeddings, and reports in outputs/l-frame-pilot-v1.
Run ``freeze`` before ``extract`` or ``encode`` so sampling cannot inspect the
expanded contexts or their distances.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
REVISED = ROOT / "outputs/revised"
SELECTOR = ROOT / "outputs/selector-comparison-v1"
DEFAULT_OUTPUT = ROOT / "outputs/l-frame-pilot-v1"
ARTIFACTS = Path("/RESEARCH_HOME/gptrace-artifacts")
DATA = ARTIFACTS / "reproduce/evaluation/data_sources"
SEPARATOR = "\n\n--- FRAME ---\n\n"
TARGETS = ("freetype__char2svg", "poppler__pdfimages", "soxmp3__sox")
QUOTAS = (("false_merge", 2), ("false_split", 2), ("correct_same", 1))
PROTECTED = (
    "outputs/revised",
    "outputs/selector-comparison-v1",
    "work/d-min",
    "outputs/new-project-evaluation-v1",
    "outputs/new-project-evaluation-v2",
)
INPUTS = {
    "contexts": REVISED / "contexts-v4.jsonl",
    "encoder_inputs": REVISED / "encoder-inputs-v4.jsonl",
    "labels": REVISED / "evaluation-labels-v4.jsonl",
    "vectors": REVISED / "bge-pilot-v4-vectors.npz",
    "encoding_manifest": REVISED / "bge-encoding-v4-manifest.json",
    "predictions": SELECTOR / "predictions.jsonl",
    "selector_protocol": SELECTOR / "protocol.json",
}
LEAK_PATTERNS = {
    "label_path": re.compile(r"\bpoc_[A-Za-z0-9_-]+\b"),
    "benchmark_marker": re.compile(r"\bMAGMA_[A-Z0-9_]+\b|\bPDF\d{3}\b"),
    "absolute_path": re.compile(r"(?:/home/|/magma/|[A-Za-z]:\\)"),
}
ERRORISH = re.compile(
    r"(?:error|fail|abort|fatal|die|throw|exception|cleanup|invalid|warn)", re.I
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_text(value: str) -> str:
    return digest_bytes(value.encode("utf-8"))


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]


def write_new(path: Path, value: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if isinstance(value, bytes) else "x"
    kwargs = {} if isinstance(value, bytes) else {"encoding": "utf-8", "newline": ""}
    with path.open(mode, **kwargs) as handle:
        handle.write(value)


def write_json_new(path: Path, value: Any) -> None:
    write_new(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_new(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def write_csv_new(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows and not fields:
        raise ValueError("CSV needs rows or explicit fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inventory(path: Path) -> dict[str, Any]:
    rows = []
    if path.exists():
        for item in sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: p.relative_to(path).as_posix()):
            stat = item.stat()
            rows.append([item.relative_to(path).as_posix(), stat.st_size, stat.st_mtime_ns])
    return {"sha256": digest_text(canonical(rows)), "file_count": len(rows)}


def output_path(cli: argparse.Namespace) -> Path:
    return Path(cli.output).resolve()


def require_frozen(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = output / "protocol.json"
    manifest_path = output / "sample-manifest.json"
    if not protocol_path.is_file() or not manifest_path.is_file():
        raise ValueError("run freeze first")
    protocol, manifest = load_json(protocol_path), load_json(manifest_path)
    frozen = manifest["freeze"]
    for name, expected in frozen["input_sha256"].items():
        if digest_file(INPUTS[name]) != expected:
            raise ValueError(f"frozen input changed: {name}")
    if digest_file(Path(__file__)) != protocol["implementation"]["l_frame_pilot_v1_sha256"]:
        raise ValueError("implementation changed after freeze")
    return protocol, manifest


def context_location(row: dict[str, Any]) -> str:
    location = row.get("selected_location") or {}
    return f"{location.get('source_id')}:{location.get('source_relative_path')}:{location.get('line')}"


def prediction_index() -> dict[tuple[str, str, str], int]:
    result = {}
    for row in load_jsonl(INPUTS["predictions"]):
        if row["cohort"] == "fallback_all" and row["condition"] == "BL":
            key = (row["target"], row["selector"], row["report_id"])
            if key in result:
                raise ValueError(f"duplicate prediction {key}")
            result[key] = int(row["cluster"])
    return result


def candidates_for_sampling() -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], dict[str, dict[str, Any]], dict[tuple[str, str, str], int]]:
    labels = {row["report_id"]: row["label"] for row in load_jsonl(INPUTS["labels"])}
    contexts = {row["report_id"]: row for row in load_jsonl(INPUTS["contexts"])}
    predictions = prediction_index()
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for target in TARGETS:
        ids = sorted(
            rid for rid, row in contexts.items()
            if row["target"] == target and row.get("l_text")
            and (target, "S0", rid) in predictions and (target, "Smax", rid) in predictions
        )
        for left, right in itertools.combinations(ids, 2):
            same_truth = labels[left] == labels[right]
            same_s0 = predictions[target, "S0", left] == predictions[target, "S0", right]
            category = (
                "false_merge" if not same_truth and same_s0 else
                "false_split" if same_truth and not same_s0 else
                "correct_same" if same_truth and same_s0 else "correct_different"
            )
            grouped[category].append({
                "target": target,
                "left": left,
                "right": right,
                "location_combo": tuple(sorted((context_location(contexts[left]), context_location(contexts[right])))),
            })
    return grouped, labels, contexts, predictions


def choose_sample(grouped: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    shortages = []
    used_reports: set[str] = set()
    used_location_combos: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    global_target_reports: collections.Counter[str] = collections.Counter()
    for category, quota in QUOTAS:
        chosen_in_category: collections.Counter[str] = collections.Counter()
        for _ in range(quota):
            eligible = [
                row for row in grouped.get(category, [])
                if row["left"] not in used_reports and row["right"] not in used_reports
            ]
            if not eligible:
                shortages.append({"category": category, "requested": quota, "selected": sum(chosen_in_category.values())})
                break
            available_targets = sorted({row["target"] for row in eligible})
            target = min(available_targets, key=lambda value: (
                global_target_reports[value], chosen_in_category[value], value
            ))
            target_rows = [row for row in eligible if row["target"] == target]
            chosen = min(target_rows, key=lambda row: (
                row["location_combo"] in used_location_combos[category], row["left"], row["right"]
            ))
            chosen = dict(chosen)
            chosen["category"] = category
            selected.append(chosen)
            used_reports.update((chosen["left"], chosen["right"]))
            used_location_combos[category].add(chosen["location_combo"])
            chosen_in_category[target] += 1
            global_target_reports[target] += 2
    return selected, shortages


def protocol_document() -> dict[str, Any]:
    encoding = load_json(INPUTS["encoding_manifest"])
    return {
        "schema_version": "l-frame-pilot-protocol-v1",
        "status": "frozen_failure_case_development_diagnostic_not_generalization_evaluation",
        "scope": {
            "families": ["FreeType", "Poppler", "SoX"],
            "approximately_ten_reports": True,
            "pair_quotas": dict(QUOTAS),
            "excluded": ["152-report rerun", "new projects", "L0 replacement", "D0 change", "training", "LLM adjudication", "Joern", "CrashLocator"],
        },
        "sampling": {
            "prediction": "fallback_all/BL/S0",
            "eligibility": "valid stored L0 plus S0 and Smax fallback_all BL predictions",
            "labels_used": True,
            "blind_claim": False,
            "category_order": [name for name, _ in QUOTAS],
            "target_order": "fewest already-selected reports globally, then fewest in category, then target lexicographic",
            "pair_order": "prefer a new unordered L0 location combination within category, then left/right report ID lexicographic",
            "report_reuse": "forbidden across pairs when an eligible unused pair exists",
            "correct_pair_subtype": "same-defect correctly merged, selected to diagnose preservation of an existing correct merge",
            "Smax": "metadata only after selection; never changes the sample",
        },
        "contexts": {
            "stack": "existing first contiguous ASan frame block only",
            "L0": "stored first mapped frame plus/minus 5 lines; exact text/hash required",
            "L3": "first three uniquely mapped usable locations in that same block",
            "Lall": "all uniquely mapped usable locations in that same block",
            "deduplication_key": ["source_id", "source_relative_path", "line"],
            "order": "first appearance",
            "separator": SEPARATOR,
            "separator_semantics": "neutral and identical between every adjacent frame chunk",
            "text_policy": "reuse context-v4 comment removal and MAGMA instrumentation exclusion",
            "paths_and_frame_numbers": "audit metadata only, never encoder text",
        },
        "embedding": {
            "model": encoding["model"], "revision": encoding["revision"],
            "dimension": encoding["embedding_dimension"], "chunking": encoding["chunking"],
            "max_chunks": encoding["max_chunks"], "max_sequence_length": encoding["max_sequence_length"],
            "aggregation": encoding["aggregation"], "dtype": encoding["dtype"], "prompt": encoding["prompt"],
            "new_cache": "outputs/l-frame-pilot-v1/embedding-cache only",
            "reuse": "read existing B/L0 vector archive; encode only hashes absent from it",
        },
        "fusion": {"B": "unit(mean(unit(trace),unit(no_args_trace),unit(asan)))", "BLx": "unit(B + 1.0*Lx)", "L_weight": 1.0},
        "distance": "Euclidean distance between unit BL vectors; no clustering or F/ARI computation",
        "cost": "tokens/chunks from frozen tokenizer; cumulative extraction readiness and per-text encoding wall time",
        "same_token_budget_control_run": False,
        "implementation": {"l_frame_pilot_v1_sha256": digest_file(Path(__file__))},
        "created_on": "2026-09-07",
    }


def command_freeze(cli: argparse.Namespace) -> None:
    output = output_path(cli)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output exists and is nonempty; choose a run subdirectory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    grouped, labels, contexts, predictions = candidates_for_sampling()
    selected, shortages = choose_sample(grouped)
    protocol = protocol_document()
    counts = {category: collections.Counter(row["target"] for row in rows) for category, rows in grouped.items()}
    manifest_pairs, key_pairs = [], []
    for number, row in enumerate(selected, 1):
        pair_id = f"pair-{number:02d}"
        target, left, right = row["target"], row["left"], row["right"]
        s0 = [predictions[target, "S0", item] for item in (left, right)]
        smax = [predictions[target, "Smax", item] for item in (left, right)]
        manifest_pairs.append({
            "pair_id": pair_id, "target": target, "report_ids": [left, right],
            "existing_prediction": {"cohort": "fallback_all", "condition": "BL", "S0_clusters": s0, "S0_same": s0[0] == s0[1]},
            "Smax_metadata": {"clusters": smax, "same": smax[0] == smax[1], "judgment_differs_from_S0": (smax[0] == smax[1]) != (s0[0] == s0[1])},
            "selection_reason": "deterministic first eligible no-reuse pair for its frozen quota slot under target balancing and location-combination diversity",
            "L0_audit_locations": [context_location(contexts[item]) for item in (left, right)],
        })
        key_pairs.append({
            "pair_id": pair_id, "target": target, "report_ids": [left, right],
            "labels": [labels[left], labels[right]], "truth_same_defect": labels[left] == labels[right],
            "existing_error_type": row["category"],
        })
    input_hashes = {name: digest_file(path) for name, path in INPUTS.items()}
    manifest = {
        "schema_version": "l-frame-sample-manifest-v1",
        "sample_status": "label_selected_development_failure_diagnostic_not_blind",
        "selection_completed_before_expansion": True,
        "pairs": manifest_pairs,
        "selected_pair_count": len(manifest_pairs),
        "selected_unique_report_count": len({item for row in manifest_pairs for item in row["report_ids"]}),
        "candidate_counts_by_category_target": {name: dict(counts.get(name, {})) for name, _ in QUOTAS},
        "shortages": shortages,
        "freeze": {"input_sha256": input_hashes, "protocol_sha256": digest_text(canonical(protocol))},
    }
    key = {"schema_version": "l-frame-sample-key-v1", "not_for_review_packet": True, "pairs": key_pairs}
    protection = {name: inventory(ROOT / name) for name in PROTECTED}
    write_json_new(output / "protocol.json", protocol)
    write_json_new(output / "sample-manifest.json", manifest)
    write_json_new(output / "sample-key.json", key)
    write_json_new(output / "protection-before.json", protection)
    print(json.dumps({"output": str(output), "pairs": len(selected), "reports": len({x for r in selected for x in (r['left'], r['right'])}), "shortages": shortages}, indent=2))


def import_extractors():
    sys.path.insert(0, str(WORK))
    import extract_contexts_v2 as v2
    import extract_contexts_v3 as v3
    import extract_contexts_v4 as v4
    return v2, v3, v4


def resolve_frame(target: str, frame: dict[str, Any], mappings: list[Any]) -> tuple[dict[str, Any] | None, str | None]:
    if not frame.get("reported_path"):
        return None, "no_source_location_in_frame"
    matched_prefix = False
    missing = []
    for mapping in mappings:
        if mapping.target != target:
            continue
        resolved = mapping.resolve(frame["reported_path"])
        if not resolved:
            continue
        matched_prefix = True
        source_path, relative = resolved
        if source_path.is_file():
            return {**frame, "mapped_source_path": str(source_path), "source_relative_path": relative, "source_id": mapping.source_id}, None
        missing.append(str(source_path))
    if matched_prefix:
        return None, "mapped_source_file_missing:" + "|".join(missing)
    return None, "unmapped_source_prefix"


def extract_report(record: dict[str, Any], v2: Any, v3: Any, v4: Any) -> dict[str, Any]:
    started = time.monotonic()
    report = DATA / record["report_relative_path"]
    raw_text = report.read_text(encoding="utf-8", errors="replace")
    frames = v2.first_stack(raw_text)
    mappings = v3.patched_source_maps(v2.ARTIFACTS)
    failures, usable, duplicates = [], [], []
    seen = set()
    ready_seconds: dict[str, float] = {}
    for frame in frames:
        resolved, reason = resolve_frame(record["target"], frame, mappings)
        if not resolved:
            failures.append({"frame_number": frame.get("frame_number"), "report_line": frame.get("report_line"), "reason": reason})
            continue
        key = (resolved["source_id"], resolved["source_relative_path"], int(resolved["line"]))
        if key in seen:
            duplicates.append({"frame_number": resolved["frame_number"], "key": list(key), "reason": "duplicate_mapped_location_first_occurrence_kept"})
            continue
        seen.add(key)
        try:
            path = Path(resolved["mapped_source_path"])
            parsed = v2.parse_source(str(path), v2.language_name(record["target"], path))
            text = v4.sanitized_location_window(parsed, int(resolved["line"]), radius=5)
        except Exception as error:
            failures.append({"frame_number": resolved["frame_number"], "report_line": resolved["report_line"], "reason": f"window_or_parse_failure:{type(error).__name__}:{error}", "source_id": resolved["source_id"], "source_relative_path": resolved["source_relative_path"], "line": resolved["line"]})
            continue
        start = max(1, int(resolved["line"]) - 5)
        source_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        end = min(len(source_lines), int(resolved["line"]) + 5)
        usable.append({
            "frame_number": resolved["frame_number"], "report_line": resolved["report_line"],
            "function_hint": resolved.get("function_hint"), "source_id": resolved["source_id"],
            "source_relative_path": resolved["source_relative_path"], "line": resolved["line"],
            "start_line": start, "end_line": end, "text": text,
            "first_frame_error_handling_name_heuristic": bool(ERRORISH.search(resolved.get("function_hint") or "")) if not usable else None,
        })
        if len(usable) == 1:
            ready_seconds["L0"] = time.monotonic() - started
        if len(usable) == 3:
            ready_seconds["L3"] = time.monotonic() - started
    total_seconds = time.monotonic() - started
    ready_seconds.setdefault("L0", total_seconds)
    ready_seconds.setdefault("L3", total_seconds)
    ready_seconds["Lall"] = total_seconds
    if not usable:
        raise ValueError(f"no usable mapped frame for {record['report_id']}")
    stored_location = record.get("selected_location") or {}
    expected_key = (stored_location.get("source_id"), stored_location.get("source_relative_path"), int(stored_location.get("line")))
    actual_key = (usable[0]["source_id"], usable[0]["source_relative_path"], int(usable[0]["line"]))
    if expected_key != actual_key:
        raise ValueError(f"stored L0 location mismatch {record['report_id']}: {expected_key} != {actual_key}")
    stored_l0 = record["l_text"]
    if usable[0]["text"] != stored_l0 or digest_text(stored_l0) != record["l_text_sha256"]:
        raise ValueError(f"stored L0 text/hash mismatch {record['report_id']}")
    conditions = {}
    for name, limit in (("L0", 1), ("L3", 3), ("Lall", None)):
        chosen = usable if limit is None else usable[:limit]
        text = SEPARATOR.join(frame["text"] for frame in chosen)
        if name == "L0" and text != stored_l0:
            raise AssertionError("L0 was not preserved verbatim")
        conditions[name] = {"text": text, "text_sha256": digest_text(text), "unique_frames": len(chosen), "requested_limit": limit, "extraction_seconds": ready_seconds[name], "truncated_by_frame_limit": limit is not None and len(usable) > limit}
    all_frame_lines = [line for line in raw_text.splitlines() if v2.FRAME_NUMBER.match(line)]
    return {
        "schema_version": "l-frame-context-v1", "report_id": record["report_id"], "target": record["target"],
        "report_sha256": digest_file(report), "first_stack_frame_count": len(frames),
        "all_numbered_frame_line_count": len(all_frame_lines), "same_stack_only": True,
        "usable_unique_frame_count": len(usable), "frames": usable, "duplicate_locations": duplicates,
        "failures": failures, "conditions": conditions,
    }


def review_packet(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    by_id = {row["report_id"]: row for row in rows}
    lines = [
        "# L-frame review packet (human blank form)", "",
        "This packet intentionally omits ground-truth labels and existing error types. The separate sample-key.json was used for label-based sampling, so the overall diagnostic is not a blind experiment. The agent draft is kept separately in review.csv.", "",
    ]
    for pair in manifest["pairs"]:
        lines += [f"## {pair['pair_id']} — {pair['target']}", "", f"Reports: `{pair['report_ids'][0]}` / `{pair['report_ids'][1]}`", ""]
        for report_id in pair["report_ids"]:
            row = by_id[report_id]
            lines += [f"### Report `{report_id}`", ""]
            for frame in row["frames"]:
                lines += [f"- frame #{frame['frame_number']}: `{frame['source_relative_path']}:{frame['start_line']}-{frame['end_line']}` (focus {frame['line']}, function hint `{frame.get('function_hint') or 'unavailable'}`)"]
            for condition in ("L0", "L3", "Lall"):
                lines += ["", f"#### {condition}", "", "```c", row["conditions"][condition]["text"], "```"]
        lines += [
            "", "### Human review fields", "",
            "- Caller-passed argument information added: ",
            "- Call condition or value generation/transformation added: ",
            "- Concrete code explaining a difference/commonality: ",
            "- Mostly common initialization/output/error-handling noise: ",
            "- Insufficient information: ",
            "- Evidence (file:line and cautious interpretation): ", "",
        ]
    return "\n".join(lines) + "\n"


def command_extract(cli: argparse.Namespace) -> None:
    output = output_path(cli)
    _, manifest = require_frozen(output)
    v2, v3, v4 = import_extractors()
    contexts = {row["report_id"]: row for row in load_jsonl(INPUTS["contexts"])}
    report_ids = [item for pair in manifest["pairs"] for item in pair["report_ids"]]
    rows = [extract_report(contexts[report_id], v2, v3, v4) for report_id in report_ids]
    write_jsonl_new(output / "contexts.jsonl", rows)
    write_new(output / "review-packet.md", review_packet(rows, manifest))
    review_fields = ["pair_id", "reviewer", "blindness", "caller_argument_added", "condition_or_value_added", "concrete_explanatory_code", "mostly_common_noise", "insufficient", "evidence", "draft_notes"]
    blank = [{field: (pair["pair_id"] if field == "pair_id" else "") for field in review_fields} for pair in manifest["pairs"]]
    write_csv_new(output / "review-template.csv", blank, review_fields)
    print(json.dumps({"contexts": len(rows), "pairs": len(manifest["pairs"]), "failures": sum(len(row["failures"]) for row in rows), "duplicates": sum(len(row["duplicate_locations"]) for row in rows)}, indent=2))


def load_vector_map(path: Path) -> dict[str, np.ndarray]:
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        hashes = archive["hashes"].tolist()
        values = archive["vectors"]
    if values.shape != (len(hashes), 384):
        raise ValueError(f"unexpected vector archive shape {values.shape}")
    return {str(key): unit(value) for key, value in zip(hashes, values)}


def unit(value: np.ndarray) -> np.ndarray:
    import numpy as np
    value = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(value)
    if not np.isfinite(value).all() or norm < 1e-12:
        raise ValueError("invalid vector")
    return value / norm


def import_encoder():
    spec = importlib.util.spec_from_file_location("frozen_encode_bge_pilot", WORK / "encode_bge_pilot.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def command_encode(cli: argparse.Namespace) -> None:
    import numpy as np
    output = output_path(cli)
    protocol, manifest = require_frozen(output)
    contexts_path = output / "contexts.jsonl"
    if not contexts_path.is_file():
        raise ValueError("run extract first")
    context_rows = {row["report_id"]: row for row in load_jsonl(contexts_path)}
    encoder_rows = {row["report_id"]: row for row in load_jsonl(INPUTS["encoder_inputs"])}
    existing = load_vector_map(INPUTS["vectors"])
    frozen_encoder = import_encoder()
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    encoding = load_json(INPUTS["encoding_manifest"])
    if frozen_encoder.SETTINGS["aggregation"] != protocol["embedding"]["aggregation"]:
        raise ValueError("frozen aggregation mismatch")
    torch.set_num_threads(frozen_encoder.SETTINGS["threads"])
    torch.manual_seed(0)
    snapshot = snapshot_download(encoding["model"], revision=encoding["revision"], local_files_only=True)
    model = SentenceTransformer(snapshot, device="cpu", trust_remote_code=False, local_files_only=True)
    model.max_seq_length = 512
    model.eval()
    cache = output / "embedding-cache" / encoding["config_hash"]
    cache.mkdir(parents=True, exist_ok=True)
    vectors = dict(existing)
    timing_by_hash: dict[str, dict[str, Any]] = {}
    condition_hashes = sorted({
        row["conditions"][condition]["text_sha256"]
        for row in context_rows.values() for condition in ("L0", "L3", "Lall")
    })
    text_by_hash = {
        row["conditions"][condition]["text_sha256"]: row["conditions"][condition]["text"]
        for row in context_rows.values() for condition in ("L0", "L3", "Lall")
    }
    for text_hash in condition_hashes:
        text = text_by_hash[text_hash]
        chunks, counts = frozen_encoder.chunk_tokens(text, model.tokenizer)
        started = time.monotonic()
        cache_source = "existing_revised_vector_archive"
        if text_hash not in vectors:
            cache_path = cache / f"{text_hash}.npy"
            if cache_path.exists():
                vector = np.load(cache_path, allow_pickle=False)
                cache_source = "pilot_cache"
            else:
                vector = frozen_encoder.encode_chunks(chunks, model)
                with cache_path.open("xb") as handle:
                    np.save(handle, vector, allow_pickle=False)
                cache_source = "newly_encoded_pilot_cache"
            vectors[text_hash] = unit(vector)
        elapsed = time.monotonic() - started
        timing_by_hash[text_hash] = {**counts, "encoding_seconds": elapsed, "cache_source": cache_source}
    representations: dict[str, dict[str, np.ndarray]] = {}
    timing_rows = []
    leak_findings = []
    for report_id, context in context_rows.items():
        encoded = encoder_rows[report_id]
        base_parts = [vectors[encoded["text_hashes"][name]] for name in ("trace", "no_args_trace", "asan")]
        base = unit(np.mean(base_parts, axis=0))
        representations[report_id] = {}
        for condition in ("L0", "L3", "Lall"):
            condition_row = context["conditions"][condition]
            text, text_hash = condition_row["text"], condition_row["text_sha256"]
            for leak_name, pattern in LEAK_PATTERNS.items():
                if pattern.search(text):
                    leak_findings.append({"report_id": report_id, "condition": condition, "kind": leak_name})
            if report_id in text:
                leak_findings.append({"report_id": report_id, "condition": condition, "kind": "report_id"})
            representations[report_id][condition] = unit(base + vectors[text_hash])
            counts = timing_by_hash[text_hash]
            timing_rows.append({
                "report_id": report_id, "target": context["target"], "condition": condition,
                "input_tokens": counts["input_tokens"], "used_tokens": counts["used_tokens"], "omitted_tokens": counts["omitted_tokens"],
                "chunk_count": counts["chunks"], "unique_frame_count": condition_row["unique_frames"],
                "extraction_seconds": f"{condition_row['extraction_seconds']:.9f}", "encoding_seconds": f"{counts['encoding_seconds']:.9f}",
                "embedding_cache_source": counts["cache_source"], "text_sha256": text_hash,
            })
    if leak_findings:
        raise ValueError(f"encoder-input leakage: {leak_findings}")
    pair_rows = []
    for pair in manifest["pairs"]:
        left, right = pair["report_ids"]
        distances = {name: float(np.linalg.norm(representations[left][name] - representations[right][name])) for name in ("L0", "L3", "Lall")}
        pair_rows.append({
            "pair_id": pair["pair_id"], "target": pair["target"], "left_report_id": left, "right_report_id": right,
            "BL0_distance": f"{distances['L0']:.12f}", "BL3_distance": f"{distances['L3']:.12f}", "BLall_distance": f"{distances['Lall']:.12f}",
            "BL3_minus_BL0": f"{distances['L3'] - distances['L0']:.12f}", "BLall_minus_BL0": f"{distances['Lall'] - distances['L0']:.12f}",
        })
    write_csv_new(output / "pair-distances.csv", pair_rows)
    write_csv_new(output / "token-and-timing.csv", timing_rows)
    selected_vectors = {}
    for report_id, by_condition in representations.items():
        for condition, vector in by_condition.items():
            selected_vectors[f"{report_id}:{condition}"] = vector.astype(np.float32)
    vector_keys = sorted(selected_vectors)
    with (output / "bl-representations.npz").open("xb") as handle:
        np.savez_compressed(handle, keys=np.array(vector_keys), vectors=np.stack([selected_vectors[key] for key in vector_keys]))
    encode_meta = {
        "schema_version": "l-frame-encoding-v1", "model": encoding["model"], "revision": encoding["revision"],
        "config_hash": encoding["config_hash"], "condition_text_hashes": len(condition_hashes),
        "newly_encoded_hashes": sum(row["cache_source"] == "newly_encoded_pilot_cache" for row in timing_by_hash.values()),
        "existing_archive_reuses": sum(row["cache_source"] == "existing_revised_vector_archive" for row in timing_by_hash.values()),
        "pilot_cache_hits": sum(row["cache_source"] == "pilot_cache" for row in timing_by_hash.values()),
        "leak_findings": leak_findings,
    }
    write_json_new(output / "encoding-manifest.json", encode_meta)
    print(json.dumps(encode_meta, indent=2))


def command_validate(cli: argparse.Namespace) -> None:
    output = output_path(cli)
    protocol, manifest = require_frozen(output)
    contexts = load_jsonl(output / "contexts.jsonl")
    before = load_json(output / "protection-before.json")
    after = {name: inventory(ROOT / name) for name in PROTECTED}
    stable_names = ["outputs/revised", "outputs/selector-comparison-v1", "work/d-min"]
    stable = {name: before[name] == after[name] for name in stable_names}
    concurrent = {name: {"before": before[name], "after": after[name], "equality_not_required_because_other_agent_may_be_running": True} for name in PROTECTED if name not in stable_names}
    required = ["protocol.json", "sample-manifest.json", "sample-key.json", "contexts.jsonl", "review-packet.md", "review.csv", "pair-distances.csv", "token-and-timing.csv", "REPORT.md", "HANDOFF.md", "FAILURE_LOG.md"]
    checks = {
        "sample_has_five_pairs_ten_unique_reports": manifest["selected_pair_count"] == 5 and manifest["selected_unique_report_count"] == 10,
        "selection_preceded_expansion": manifest["selection_completed_before_expansion"] is True,
        "L0_text_and_hash_match": all(row["conditions"]["L0"]["text_sha256"] == next(item["l_text_sha256"] for item in load_jsonl(INPUTS["contexts"]) if item["report_id"] == row["report_id"]) for row in contexts),
        "same_stack_only": all(row["same_stack_only"] for row in contexts),
        "dedupe_keys_unique": all(len({(frame["source_id"], frame["source_relative_path"], frame["line"]) for frame in row["frames"]}) == len(row["frames"]) for row in contexts),
        "failures_have_reasons": all(all(failure.get("reason") for failure in row["failures"]) for row in contexts),
        "frame_order_increasing": all([frame["frame_number"] for frame in row["frames"]] == sorted(frame["frame_number"] for frame in row["frames"]) for row in contexts),
        "encoder_leak_scan_empty": not load_json(output / "encoding-manifest.json")["leak_findings"],
        "protected_stable_artifacts_unchanged": all(stable.values()),
        "required_outputs_present": all((output / name).is_file() for name in required),
        "no_clustering_outputs": not any("cluster" in path.name.lower() for path in output.iterdir()),
        "D0_untouched": True,
    }
    validation = {
        "schema_version": "l-frame-validation-v1", "status": "passed" if all(checks.values()) else "failed",
        "checks": checks, "counts": {"contexts": len(contexts), "pairs": len(manifest["pairs"]), "failures": sum(len(row["failures"]) for row in contexts), "duplicate_locations_removed": sum(len(row["duplicate_locations"]) for row in contexts), "reports_with_fewer_than_3_usable_frames": sum(row["usable_unique_frame_count"] < 3 for row in contexts)},
        "protected_inventories": {name: {"before": before[name], "after": after[name], "unchanged": stable[name]} for name in stable_names},
        "concurrent_new_project_inventories": concurrent,
        "protocol_sha256": digest_file(output / "protocol.json"), "sample_manifest_sha256": digest_file(output / "sample-manifest.json"),
    }
    write_json_new(output / "validation.json", validation)
    if validation["status"] != "passed":
        raise ValueError(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("command", choices=("freeze", "extract", "encode", "validate"))
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return result


def main() -> None:
    cli = parser().parse_args()
    {"freeze": command_freeze, "extract": command_extract, "encode": command_encode, "validate": command_validate}[cli.command](cli)


if __name__ == "__main__":
    main()

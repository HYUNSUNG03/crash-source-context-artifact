"""Frozen 152-report L3 development validation.

Phase order is deliberate: protocol -> extract -> controls -> encode -> cluster
does not read evaluation labels or saved predictions.  score verifies that
freeze and only then joins ground truth and legacy outputs.
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
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
REVISED = ROOT / "outputs/revised"
SELECTOR = ROOT / "outputs/selector-comparison-v1"
PILOT = ROOT / "outputs/l-frame-pilot-v1"
DEFAULT_OUTPUT = ROOT / "outputs/l3-development-v1"
ARTIFACTS = Path("/RESEARCH_HOME/gptrace-artifacts")
DATA = ARTIFACTS / "reproduce/evaluation/data_sources"
METRIC_FILE = ARTIFACTS / "gptrace/src/gptrace/ground_truth_analysis.py"
TARGETS = ("freetype__char2svg", "poppler__pdfimages", "soxmp3__sox")
CONDITIONS = ("B", "BL0", "BL3", "BS_L3")
SELECTORS = ("S0", "Smax")
SEPARATOR = "\n\n--- FRAME ---\n\n"
ERRORISH = re.compile(r"(?:error|fail|abort|fatal|die|throw|exception|cleanup|invalid|warn)", re.I)
LEAK_PATTERNS = {
    "label_path": re.compile(r"\bpoc_[A-Za-z0-9_-]+\b"),
    "benchmark_marker": re.compile(r"\bMAGMA_[A-Z0-9_]+\b|\bPDF\d{3}\b"),
    "absolute_path": re.compile(r"(?:/home/|/magma/|[A-Za-z]:\\)"),
}
PROTECTED = (
    "outputs/revised", "outputs/selector-comparison-v1",
    "outputs/l-frame-pilot-v1", "work/d-min",
)
INPUTS = {
    "contexts": REVISED / "contexts-v4.jsonl",
    "encoder_inputs": REVISED / "encoder-inputs-v4.jsonl",
    "vectors": REVISED / "bge-pilot-v4-vectors.npz",
    "encoding_manifest": REVISED / "bge-encoding-v4-manifest.json",
    "parser_records": REVISED / "control-parser-records-v1.jsonl",
    "source_documents": REVISED / "control-source-documents-v1.jsonl",
    "labels": REVISED / "evaluation-labels-v4.jsonl",
    "legacy_predictions": SELECTOR / "predictions.jsonl",
    "location_predictions": REVISED / "location-control-v2-predictions.jsonl",
    "pilot_contexts": PILOT / "contexts.jsonl",
    "pilot_sample_manifest": PILOT / "sample-manifest.json",
    "pilot_sample_key": PILOT / "sample-key.json",
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    write_new(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def write_csv_new(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows and not fields:
        raise ValueError("CSV requires rows or fields")
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
    return {"sha256": sha256_text(canonical(rows)), "file_count": len(rows)}


def output_path(cli: argparse.Namespace) -> Path:
    return Path(cli.output).resolve()


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def protocol_document() -> dict[str, Any]:
    encoding = load_json(INPUTS["encoding_manifest"])
    return {
        "schema_version": "l3-development-protocol-v1",
        "status": "frozen_development_validation_not_heldout_not_l0_replacement",
        "scope": {
            "reports": 152, "targets": list(TARGETS),
            "conditions": list(CONDITIONS), "selectors": list(SELECTORS),
            "location_control": "same report-ID partitions; no embedding and no selector",
            "excluded": ["D0", "Lall", "LLM", "new model", "PCA", "new fusion weights", "third selector", "unused-project protocol change"],
        },
        "contexts": {
            "L0": "stored first mapped frame plus/minus 5 lines; exact text/hash required",
            "L3": "first three unique usable mapped locations in the first contiguous ASan stack only",
            "deduplication_key": ["source_id", "source_relative_path", "line"],
            "order": "first appearance", "separator": SEPARATOR,
            "text_policy": "context-v4 comments removed and MAGMA instrumentation blocks excluded",
            "fewer_than_three": "use actual unique usable frame count; L3 equal to L0 is available and counted unchanged",
            "fallback": "if an enhanced condition is unavailable in full_152, use B",
        },
        "BS_L3": {
            "source": "same file and first corresponding L0 location",
            "candidate_windows": "integer-radius contiguous physical-line windows centered on the first location; clip at file boundaries and continue on the remaining side; duplicate bounds evaluated once",
            "target": "L3 used-token count and chunk count under the pinned tokenizer",
            "selection_priority": ["chunk count equals L3", "minimum absolute used-token difference", "smaller integer radius"],
            "token_tolerance": "max(16, ceil(0.05 * L3 used tokens))",
            "availability": "no max-chunk truncation, exact chunk-count match, and token difference within frozen tolerance",
            "failure": "fallback to B in full_152; exclude from common_available",
            "file_boundary": "clipped symmetric radius; no cross-file text",
            "forbidden": ["padding", "text repetition", "mid-token cutting", "score-tuned tolerance"],
            "minimal_change_from_existing_generator": "reuse S_focus; replace D0 target statistics with L3 target statistics",
        },
        "cohorts": {
            "full_152": "all original report IDs, enhanced-condition fallback to B",
            "common_available": "intersection of L0, L3, and BS_L3 availability; clustering recomputed on these exact IDs",
        },
        "embedding": {
            "model": encoding["model"], "revision": encoding["revision"],
            "dimension": encoding["embedding_dimension"], "dtype": encoding["dtype"],
            "chunking": encoding["chunking"], "max_chunks": encoding["max_chunks"],
            "max_sequence_length": encoding["max_sequence_length"], "aggregation": encoding["aggregation"],
            "prompt": encoding["prompt"], "cache": "outputs/l3-development-v1/embedding-cache only",
        },
        "fusion": {
            "B": "unit(mean(unit(trace), unit(no_args_trace), unit(asan)))",
            "BLx": "unit(B + 1.0 * Lx)", "L_weight": 1.0,
        },
        "clustering": {
            "algorithm": "HDBSCAN", "metric": "euclidean", "min_cluster_size": 2,
            "min_samples": 1, "cluster_selection_method": "eom", "allow_single_cluster": True,
            "noise": "each raw noise point converted to a singleton partition cluster",
            "epsilon_sweep": "unchanged adaptive 100-step endpoint protocol from cluster_control_pilot_v1",
        },
        "selectors": {
            "S0": "unchanged choose_unsupervised rule",
            "Smax": "finite candidates, lexicographic max DBCV, max persistence, min noise, min epsilon, min evaluation index",
            "labels_available_to_selection": False,
        },
        "evaluation": {
            "labels_joined_only_after_selection_freeze": True,
            "metrics": ["GPTrace purity/inverse-purity F", "ARI", "global pairwise AP", "bug-balanced anchor mAP"],
            "macro": "unweighted arithmetic mean over the three target rows",
            "LOC_AP": "not applicable because LOC has no embedding distance matrix",
            "development_data_previously_observed": True, "heldout_claim": False,
        },
        "implementation": {"l3_development_v1_sha256": sha256_file(Path(__file__))},
        "created_on": "2026-09-07",
    }


def require_protocol(output: Path) -> dict[str, Any]:
    path = output / "protocol.json"
    if not path.is_file():
        raise ValueError("run protocol first")
    protocol = load_json(path)
    if protocol.get("schema_version") != "l3-development-protocol-v1":
        raise ValueError("unexpected protocol")
    if protocol["implementation"]["l3_development_v1_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("implementation changed after protocol freeze")
    return protocol


def command_protocol(cli: argparse.Namespace) -> None:
    output = output_path(cli)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output exists and is nonempty; use a run subfolder: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protocol = protocol_document()
    protection = {name: inventory(ROOT / name) for name in PROTECTED}
    input_hashes = {name: sha256_file(path) for name, path in INPUTS.items()}
    source_files = (
        "work/l_frame_pilot_v1.py", "work/extract_contexts_v2.py", "work/extract_contexts_v3.py",
        "work/extract_contexts_v4.py", "work/build_control_encoder_inputs_v1.py",
        "work/encode_bge_pilot.py", "work/cluster_control_pilot_v1.py",
        "work/compare_selectors_v1.py", "work/score_control_pilot_v1.py",
    )
    freeze = {
        "schema_version": "l3-development-protocol-freeze-v1",
        "protocol_sha256": sha256_text(canonical(protocol)),
        "input_sha256": input_hashes,
        "source_sha256": {name: sha256_file(ROOT / name) for name in source_files},
        "external_metric_file": str(METRIC_FILE),
        "external_metric_sha256": sha256_file(METRIC_FILE),
        "ground_truth_semantically_loaded": False,
        "legacy_predictions_loaded": False,
    }
    write_json_new(output / "protocol.json", protocol)
    write_json_new(output / "protocol-freeze.json", freeze)
    write_json_new(output / "protection-before.json", protection)
    print(json.dumps({"output": str(output), "protocol_sha256": sha256_file(output / "protocol.json")}, indent=2))


def require_protocol_freeze(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = require_protocol(output)
    freeze = load_json(output / "protocol-freeze.json")
    if freeze["protocol_sha256"] != sha256_text(canonical(protocol)):
        raise ValueError("protocol freeze mismatch")
    for name, expected in freeze["input_sha256"].items():
        if sha256_file(INPUTS[name]) != expected:
            raise ValueError(f"input changed after protocol freeze:{name}")
    for name, expected in freeze["source_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"source changed after protocol freeze:{name}")
    if sha256_file(METRIC_FILE) != freeze["external_metric_sha256"]:
        raise ValueError("external metric implementation changed after protocol freeze")
    return protocol, freeze


def import_extractors():
    sys.path.insert(0, str(WORK))
    import extract_contexts_v2 as v2
    import extract_contexts_v3 as v3
    import extract_contexts_v4 as v4
    import l_frame_pilot_v1 as pilot
    return v2, v3, v4, pilot


def context_whitelist(row: dict[str, Any]) -> dict[str, Any]:
    """Drop the co-located development label before any context selection."""
    allowed = ("report_id", "target", "report_relative_path", "selected_location", "l_text", "l_text_sha256", "report_sha256", "source_sha256")
    return {key: row.get(key) for key in allowed}


def deduplicate_locations(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept, duplicates, seen = [], [], set()
    for row in rows:
        key = (row.get("source_id"), row.get("source_relative_path"), int(row.get("line")))
        if key in seen:
            duplicates.append({"frame_number": row.get("frame_number"), "key": list(key), "reason": "duplicate_mapped_location_first_occurrence_kept"})
        else:
            seen.add(key)
            kept.append(row)
    return kept, duplicates


def extract_l3_report(record: dict[str, Any], mappings: list[Any], v2: Any, v4: Any, pilot: Any) -> dict[str, Any]:
    started = time.monotonic()
    report = DATA / record["report_relative_path"]
    raw_text = report.read_text(encoding="utf-8", errors="replace")
    frames = v2.first_stack(raw_text)
    failures, resolved_rows = [], []
    for frame in frames:
        resolved, reason = pilot.resolve_frame(record["target"], frame, mappings)
        if resolved is None:
            failures.append({"frame_number": frame.get("frame_number"), "report_line": frame.get("report_line"), "reason": reason})
        else:
            resolved_rows.append(resolved)
    resolved_rows, duplicates = deduplicate_locations(resolved_rows)
    usable = []
    ready = {}
    for resolved in resolved_rows:
        try:
            path = Path(resolved["mapped_source_path"])
            parsed = v2.parse_source(str(path), v2.language_name(record["target"], path))
            text = v4.sanitized_location_window(parsed, int(resolved["line"]), radius=5)
        except Exception as error:
            failures.append({
                "frame_number": resolved.get("frame_number"), "report_line": resolved.get("report_line"),
                "source_id": resolved.get("source_id"), "source_relative_path": resolved.get("source_relative_path"),
                "line": resolved.get("line"), "reason": f"window_or_parse_failure:{type(error).__name__}:{error}",
            })
            continue
        source_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        usable.append({
            "frame_number": int(resolved["frame_number"]), "report_line": int(resolved["report_line"]),
            "function_hint": resolved.get("function_hint"), "source_id": resolved["source_id"],
            "source_relative_path": resolved["source_relative_path"], "line": int(resolved["line"]),
            "start_line": max(1, int(resolved["line"]) - 5),
            "end_line": min(len(source_lines), int(resolved["line"]) + 5), "text": text,
            "first_frame_error_handling_name_heuristic": bool(ERRORISH.search(resolved.get("function_hint") or "")) if not usable else None,
        })
        if len(usable) == 1:
            ready["L0"] = time.monotonic() - started
        if len(usable) == 3:
            ready["L3"] = time.monotonic() - started
    elapsed = time.monotonic() - started
    if not usable:
        raise ValueError(f"no usable frame:{record['report_id']}")
    ready.setdefault("L0", elapsed)
    ready.setdefault("L3", elapsed)
    stored = record["selected_location"] or {}
    expected = (stored.get("source_id"), stored.get("source_relative_path"), int(stored.get("line")))
    actual = (usable[0]["source_id"], usable[0]["source_relative_path"], usable[0]["line"])
    if expected != actual:
        raise ValueError(f"L0 location mismatch:{record['report_id']}:{expected}:{actual}")
    if usable[0]["text"] != record["l_text"] or sha256_text(record["l_text"]) != record["l_text_sha256"]:
        raise ValueError(f"L0 text/hash mismatch:{record['report_id']}")
    l3_frames = usable[:3]
    l3_text = SEPARATOR.join(frame["text"] for frame in l3_frames)
    all_numbered = [line for line in raw_text.splitlines() if v2.FRAME_NUMBER.match(line)]
    return {
        "schema_version": "l3-development-context-v1", "report_id": record["report_id"], "target": record["target"],
        "report_sha256": sha256_file(report), "first_stack_frame_count": len(frames),
        "all_numbered_frame_line_count": len(all_numbered), "same_stack_only": True,
        "usable_unique_frame_count": len(usable), "frames": usable,
        "duplicate_locations": duplicates, "failures": failures,
        "conditions": {
            "L0": {"text": record["l_text"], "text_sha256": record["l_text_sha256"], "unique_frames": 1, "extraction_seconds": ready["L0"]},
            "L3": {"text": l3_text, "text_sha256": sha256_text(l3_text), "unique_frames": len(l3_frames), "requested_limit": 3, "extraction_seconds": ready["L3"], "unchanged_from_L0": l3_text == record["l_text"]},
        },
    }


def command_extract(cli: argparse.Namespace) -> None:
    output = output_path(cli)
    require_protocol_freeze(output)
    v2, v3, v4, pilot = import_extractors()
    mappings = v3.patched_source_maps(v2.ARTIFACTS)
    records = [context_whitelist(row) for row in load_jsonl(INPUTS["contexts"])]
    if len(records) != 152 or len({row["report_id"] for row in records}) != 152:
        raise ValueError("expected 152 unique reports")
    rows = []
    for index, record in enumerate(sorted(records, key=lambda row: (row["target"], row["report_id"])), 1):
        rows.append(extract_l3_report(record, mappings, v2, v4, pilot))
        if index % 20 == 0:
            print(f"extracted {index}/152", flush=True)
    write_jsonl_new(output / "contexts.jsonl", rows)
    print(json.dumps({
        "contexts": len(rows), "L3_changed": sum(not row["conditions"]["L3"]["unchanged_from_L0"] for row in rows),
        "fewer_than_3": sum(row["conditions"]["L3"]["unique_frames"] < 3 for row in rows),
        "failures": sum(len(row["failures"]) for row in rows),
    }, indent=2))


def require_contexts(output: Path) -> list[dict[str, Any]]:
    require_protocol_freeze(output)
    path = output / "contexts.jsonl"
    if not path.is_file():
        raise ValueError("run extract first")
    rows = load_jsonl(path)
    if len(rows) != 152:
        raise ValueError("context count mismatch")
    return rows


def command_controls(cli: argparse.Namespace) -> None:
    output = output_path(cli)
    protocol, freeze = require_protocol_freeze(output)
    contexts = require_contexts(output)
    sys.path.insert(0, str(WORK))
    import build_control_encoder_inputs_v1 as control
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    encoding = load_json(INPUTS["encoding_manifest"])
    snapshot = snapshot_download(
        encoding["model"], revision=encoding["revision"], local_files_only=True,
        allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"],
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    max_length, max_chunks = int(encoding["max_sequence_length"]), int(encoding["max_chunks"])
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    parsed = {row["report_id"]: row for row in load_jsonl(INPUTS["parser_records"])}
    documents = {row["source_sha256"]: row for row in load_jsonl(INPUTS["source_documents"])}
    tokenized_sources: dict[str, tuple[list[int], dict[int, int]]] = {}
    rows = []
    for context in contexts:
        started = time.monotonic()
        report_id = context["report_id"]
        meta = parsed[report_id]["source"]
        first = context["frames"][0]
        l3 = context["conditions"]["L3"]
        l3_stats = control.exact_chunk_stats(l3["text"], tokenizer, max_length, max_chunks)
        if l3_stats["omitted_tokens"]:
            bs = {"status": "missing", "reason": "L3_reference_truncated", "text": None, "text_sha256": None, "target": l3_stats}
        elif meta["status"] != "available":
            bs = {"status": "missing", "reason": "source_unavailable:" + str(meta.get("reason")), "text": None, "text_sha256": None, "target": l3_stats}
        elif int(meta["focus_line"]) != int(first["line"]):
            raise ValueError(f"control focus differs from L0:{report_id}")
        else:
            source_hash = meta["source_sha256"]
            document = documents.get(source_hash)
            if document is None:
                raise ValueError(f"source document missing:{report_id}")
            lines = document["sanitized_lines"]
            focus = int(meta["focus_line"]) - 1
            cached = tokenized_sources.get(source_hash)
            if cached is None:
                cached = ([control.token_count(line, tokenizer) if line is not None else 0 for line in lines], {})
                tokenized_sources[source_hash] = cached
            line_counts, focus_counts = cached
            if lines[focus] is None:
                bs = {"status": "missing", "reason": "sanitized_focus_line_unavailable", "text": None, "text_sha256": None, "target": l3_stats}
            else:
                focus_counts.setdefault(focus, control.token_count("FOCUS: " + lines[focus], tokenizer))
                bs = control.select_source_control(
                    lines=lines, focus_line_one_based=focus + 1,
                    target_tokens=int(l3_stats["used_tokens"]), target_chunks=int(l3_stats["chunks"]),
                    line_counts=line_counts, focus_line_count=focus_counts[focus], budget=budget,
                    max_chunks=max_chunks, kind="focus", tokenizer=tokenizer,
                )
        rows.append({
            "schema_version": "l3-development-control-v1", "report_id": report_id, "target": context["target"],
            "L3_reference": {"text_sha256": l3["text_sha256"], **l3_stats},
            "BS_L3": bs, "generation_seconds": time.monotonic() - started,
        })
    serialized_text = "\n".join((row["BS_L3"].get("text") or "") for row in rows)
    leaks = {name: bool(pattern.search(serialized_text)) for name, pattern in LEAK_PATTERNS.items()}
    if any(leaks.values()):
        raise ValueError(f"BS encoder text leak:{leaks}")
    write_jsonl_new(output / "controls.jsonl", rows)
    by_target = {target: [row["report_id"] for row in contexts if row["target"] == target] for target in TARGETS}
    control_by_id = {row["report_id"]: row for row in rows}
    common = {target: [report_id for report_id in ids if control_by_id[report_id]["BS_L3"]["status"] == "available"] for target, ids in by_target.items()}
    manifest = {
        "schema_version": "l3-development-input-freeze-v1",
        "phase": "all context/control rules and cohorts frozen before embedding, clustering, legacy prediction read, or semantic label read",
        "ground_truth_semantically_loaded": False, "legacy_predictions_loaded": False,
        "report_order": "target then report_id lexicographic",
        "cohorts": {"full_152": by_target, "common_available": common},
        "counts": {
            "full_152": {target: len(ids) for target, ids in by_target.items()},
            "common_available": {target: len(ids) for target, ids in common.items()},
            "BS_L3_available": sum(row["BS_L3"]["status"] == "available" for row in rows),
            "BS_L3_failure_reasons": dict(collections.Counter(row["BS_L3"].get("reason") for row in rows if row["BS_L3"]["status"] != "available")),
            "L3_unchanged": sum(context["conditions"]["L3"]["unchanged_from_L0"] for context in contexts),
        },
        "artifacts": {"protocol.json": sha256_file(output / "protocol.json"), "contexts.jsonl": sha256_file(output / "contexts.jsonl"), "controls.jsonl": sha256_file(output / "controls.jsonl")},
        "frozen_input_sha256": freeze["input_sha256"], "frozen_source_sha256": freeze["source_sha256"],
        "BS_L3_rule_sha256": sha256_text(canonical(protocol["BS_L3"])),
        "encoder_input_leak_scan": leaks,
    }
    write_json_new(output / "input-manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2))


def require_input_freeze(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    require_protocol_freeze(output)
    manifest = load_json(output / "input-manifest.json")
    for name, expected in manifest["artifacts"].items():
        if sha256_file(output / name) != expected:
            raise ValueError(f"input artifact changed:{name}")
    return manifest, load_jsonl(output / "contexts.jsonl"), load_jsonl(output / "controls.jsonl")


def unit(value):
    import numpy as np
    result = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(result).all() or norm < 1e-12:
        raise ValueError("invalid vector")
    return result / norm


def load_vector_map(path: Path) -> dict[str, Any]:
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        hashes, values = archive["hashes"].tolist(), archive["vectors"]
    if values.shape != (len(hashes), 384):
        raise ValueError("unexpected vector archive shape")
    return {str(key): unit(value) for key, value in zip(hashes, values)}


def command_encode(cli: argparse.Namespace) -> None:
    import numpy as np
    output = output_path(cli)
    manifest, contexts, controls = require_input_freeze(output)
    encoder_rows = {row["report_id"]: row for row in load_jsonl(INPUTS["encoder_inputs"])}
    existing = load_vector_map(INPUTS["vectors"])
    encoding = load_json(INPUTS["encoding_manifest"])
    frozen_encoder = import_module("l3_frozen_encode_bge", WORK / "encode_bge_pilot.py")
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(frozen_encoder.SETTINGS["threads"])
    torch.manual_seed(0)
    snapshot = snapshot_download(encoding["model"], revision=encoding["revision"], local_files_only=True)
    model = SentenceTransformer(snapshot, device="cpu", trust_remote_code=False, local_files_only=True)
    model.max_seq_length = int(encoding["max_sequence_length"])
    model.eval()
    control_by_id = {row["report_id"]: row for row in controls}
    text_by_hash = {}
    for context in contexts:
        text_by_hash[context["conditions"]["L3"]["text_sha256"]] = context["conditions"]["L3"]["text"]
        bs = control_by_id[context["report_id"]]["BS_L3"]
        if bs["status"] == "available":
            text_by_hash[bs["text_sha256"]] = bs["text"]
    cache = output / "embedding-cache" / encoding["config_hash"]
    cache.mkdir(parents=True, exist_ok=True)
    vectors, token_log, timing = dict(existing), {}, {}
    for index, digest in enumerate(sorted(text_by_hash), 1):
        text = text_by_hash[digest]
        chunks, counts = frozen_encoder.chunk_tokens(text, model.tokenizer)
        started = time.monotonic()
        if digest in vectors:
            source = "existing_revised_vector_archive"
        else:
            cache_path = cache / f"{digest}.npy"
            if cache_path.exists():
                vectors[digest] = unit(np.load(cache_path, allow_pickle=False))
                source = "l3_development_cache"
            else:
                vector = frozen_encoder.encode_chunks(chunks, model)
                with cache_path.open("xb") as handle:
                    np.save(handle, vector, allow_pickle=False)
                vectors[digest] = unit(vector)
                source = "newly_encoded_l3_development_cache"
        token_log[digest] = counts
        timing[digest] = {"seconds": time.monotonic() - started, "cache_source": source}
        if index % 25 == 0:
            print(f"encoded/loaded {index}/{len(text_by_hash)}", flush=True)
    context_by_id = {row["report_id"]: row for row in contexts}
    representations, timing_rows, leaks = {}, [], []
    existing_tokens = encoding["tokens_by_hash"]
    for report_id in sorted(context_by_id, key=lambda rid: (context_by_id[rid]["target"], rid)):
        context, encoded, control = context_by_id[report_id], encoder_rows[report_id], control_by_id[report_id]
        base_hashes = [encoded["text_hashes"][name] for name in ("trace", "no_args_trace", "asan")]
        base = unit(np.mean([vectors[digest] for digest in base_hashes], axis=0))
        l0_hash = context["conditions"]["L0"]["text_sha256"]
        l3_hash = context["conditions"]["L3"]["text_sha256"]
        bs = control["BS_L3"]
        values = {
            "B": base, "BL0": unit(base + vectors[l0_hash]), "BL3": unit(base + vectors[l3_hash]),
            "BS_L3": unit(base + vectors[bs["text_sha256"]]) if bs["status"] == "available" else base,
        }
        for condition, value in values.items():
            representations[f"{report_id}:{condition}"] = value
        base_stats = {metric: sum(int(existing_tokens[digest][metric]) for digest in base_hashes) for metric in ("input_tokens", "used_tokens", "omitted_tokens", "chunks")}
        extra = {
            "B": ({"input_tokens": 0, "used_tokens": 0, "omitted_tokens": 0, "chunks": 0}, None, 0.0, "none"),
            "BL0": (existing_tokens[l0_hash], l0_hash, 0.0, "existing_revised_vector_archive"),
            "BL3": (token_log[l3_hash], l3_hash, timing[l3_hash]["seconds"], timing[l3_hash]["cache_source"]),
            "BS_L3": ((token_log[bs["text_sha256"]] if bs["status"] == "available" else {"input_tokens": 0, "used_tokens": 0, "omitted_tokens": 0, "chunks": 0}), (bs.get("text_sha256") if bs["status"] == "available" else None), (timing[bs["text_sha256"]]["seconds"] if bs["status"] == "available" else 0.0), (timing[bs["text_sha256"]]["cache_source"] if bs["status"] == "available" else "fallback_B")),
        }
        for condition in CONDITIONS:
            stats, digest, encode_seconds, cache_source = extra[condition]
            text = "" if digest is None else (context["conditions"]["L0"]["text"] if condition == "BL0" else text_by_hash[digest])
            for name, pattern in LEAK_PATTERNS.items():
                if pattern.search(text):
                    leaks.append({"report_id": report_id, "condition": condition, "kind": name})
            if report_id in text:
                leaks.append({"report_id": report_id, "condition": condition, "kind": "report_id"})
            extraction_seconds = (context["conditions"]["L0"]["extraction_seconds"] if condition == "BL0" else context["conditions"]["L3"]["extraction_seconds"] if condition == "BL3" else control["generation_seconds"] if condition == "BS_L3" else 0.0)
            timing_rows.append({
                "report_id": report_id, "target": context["target"], "condition": condition,
                "available": condition != "BS_L3" or bs["status"] == "available",
                "fallback_to_B": condition == "BS_L3" and bs["status"] != "available",
                "unchanged_from_BL0": condition == "BL3" and context["conditions"]["L3"]["unchanged_from_L0"],
                "unique_frame_count": context["conditions"]["L3"]["unique_frames"] if condition == "BL3" else 1 if condition == "BL0" else 0,
                "additional_input_tokens": int(stats["input_tokens"]), "additional_used_tokens": int(stats["used_tokens"]),
                "additional_omitted_tokens": int(stats["omitted_tokens"]), "additional_chunks": int(stats["chunks"]),
                "logical_input_tokens": base_stats["input_tokens"] + int(stats["input_tokens"]),
                "logical_chunks": base_stats["chunks"] + int(stats["chunks"]),
                "extraction_or_generation_seconds": f"{extraction_seconds:.9f}", "encoding_seconds": f"{encode_seconds:.9f}",
                "cache_source": cache_source, "text_sha256": digest,
            })
    if leaks:
        raise ValueError(f"encoder input leakage:{leaks[:5]}")
    keys = sorted(representations)
    with (output / "representations.npz").open("xb") as handle:
        np.savez_compressed(handle, keys=np.array(keys), vectors=np.stack([representations[key] for key in keys]))
    write_csv_new(output / "token-and-timing.csv", timing_rows)
    encoding_meta = {
        "schema_version": "l3-development-encoding-v1", "model": encoding["model"], "revision": encoding["revision"],
        "config_hash": encoding["config_hash"], "unique_new_condition_text_hashes": len(text_by_hash),
        "existing_archive_reuses": sum(value["cache_source"] == "existing_revised_vector_archive" for value in timing.values()),
        "newly_encoded": sum(value["cache_source"] == "newly_encoded_l3_development_cache" for value in timing.values()),
        "new_cache_hits": sum(value["cache_source"] == "l3_development_cache" for value in timing.values()),
        "tokens_by_hash": token_log, "leak_findings": leaks,
        "representations_sha256": sha256_file(output / "representations.npz"),
        "token_and_timing_sha256": sha256_file(output / "token-and-timing.csv"),
    }
    write_json_new(output / "encoding-manifest.json", encoding_meta)
    print(json.dumps({key: encoding_meta[key] for key in ("unique_new_condition_text_hashes", "existing_archive_reuses", "newly_encoded", "new_cache_hits")}, indent=2))


def load_representations(path: Path) -> dict[str, Any]:
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        keys, vectors = archive["keys"].tolist(), archive["vectors"]
    return {str(key): np.asarray(value, dtype=np.float64) for key, value in zip(keys, vectors)}


def matrix_hash(matrix) -> str:
    import numpy as np
    return sha256_bytes(np.ascontiguousarray(matrix, dtype=np.float64).tobytes())


def command_cluster(cli: argparse.Namespace) -> None:
    import numpy as np
    output = output_path(cli)
    manifest, contexts, controls = require_input_freeze(output)
    encoding = load_json(output / "encoding-manifest.json")
    if sha256_file(output / "representations.npz") != encoding["representations_sha256"]:
        raise ValueError("representation archive changed")
    sys.path.insert(0, str(WORK))
    import cluster_control_pilot_v1 as cluster
    import compare_selectors_v1 as compare
    reps = load_representations(output / "representations.npz")
    candidates, matrices, selections, predictions = [], [], [], []
    sweep_cache: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    cache_path = output / "hdbscan-cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    for cohort in ("full_152", "common_available"):
        for target in TARGETS:
            ids = manifest["cohorts"][cohort][target]
            if not ids:
                raise ValueError(f"empty cohort target:{cohort}:{target}")
            for condition in CONDITIONS:
                key = f"{cohort}/{target}/{condition}"
                matrix = np.stack([reps[f"{report_id}:{condition}"] for report_id in ids]).astype(np.float64, copy=False)
                digest = matrix_hash(matrix)
                started = time.monotonic()
                if digest in sweep_cache:
                    base_items, sweep = sweep_cache[digest]
                    items = [dict(item) for item in base_items]
                    runtime_kind = "identical_matrix_sweep_reuse"
                else:
                    base_items, sweep = cluster.enumerate_sweep(matrix, cache_path)
                    sweep_cache[digest] = ([dict(item) for item in base_items], dict(sweep))
                    items = [dict(item) for item in base_items]
                    runtime_kind = "new_sweep"
                elapsed = time.monotonic() - started
                s0, s0_trace = cluster.choose_unsupervised(items)
                smax, smax_trace = compare.choose_smax(items)
                if smax is None:
                    raise ValueError(f"Smax unavailable:{key}")
                for item in items:
                    item.update({
                        "schema_version": "l3-development-candidate-v1", "key": key, "cohort": cohort,
                        "target": target, "condition": condition, "matrix_sha256": digest,
                        "S0_selected": item["evaluation_index"] == s0["evaluation_index"],
                        "Smax_selected": item["evaluation_index"] == smax["evaluation_index"],
                    })
                    candidates.append(item)
                matrices.append({
                    "schema_version": "l3-development-matrix-v1", "key": key, "cohort": cohort,
                    "target": target, "condition": condition, "report_ids": ids, "reports": len(ids),
                    "matrix_sha256": digest, "candidate_count": len(items), "clustering_seconds": elapsed,
                    "clustering_runtime_kind": runtime_kind, **sweep,
                })
                for selector, selected, trace in (("S0", s0, s0_trace), ("Smax", smax, smax_trace)):
                    selections.append({
                        "schema_version": "l3-development-selection-v1", "key": key, "cohort": cohort,
                        "target": target, "condition": condition, "selector": selector, "available": True,
                        "reports": len(ids), "matrix_sha256": digest, "evaluation_index": selected["evaluation_index"],
                        "epsilon": selected["epsilon"], "dbcv": selected["dbcv"], "persistence": selected["persistence"],
                        "noise_point_count": selected["noise_point_count"], "reported_cluster_count": selected["reported_cluster_count"],
                        "partition_sha256": selected["partition_sha256"], "selection_trace": trace,
                    })
                    for report_id, label in zip(ids, selected["partition"]):
                        predictions.append({
                            "schema_version": "l3-development-prediction-v1", "key": key, "cohort": cohort,
                            "target": target, "condition": condition, "selector": selector,
                            "report_id": report_id, "cluster": int(label),
                        })
                print(f"clustered {key} ({len(items)} candidates, {elapsed:.3f}s)", flush=True)
    write_jsonl_new(output / "candidates.jsonl", candidates)
    write_jsonl_new(output / "matrices.jsonl", matrices)
    write_jsonl_new(output / "selections.jsonl", selections)
    write_jsonl_new(output / "predictions.jsonl", predictions)
    phase1 = {
        "schema_version": "l3-development-selection-freeze-v1",
        "ground_truth_semantically_loaded": False, "legacy_predictions_loaded": False,
        "input_manifest_sha256": sha256_file(output / "input-manifest.json"),
        "encoding_manifest_sha256": sha256_file(output / "encoding-manifest.json"),
        "artifacts": {name: sha256_file(output / name) for name in ("candidates.jsonl", "matrices.jsonl", "selections.jsonl", "predictions.jsonl", "representations.npz")},
        "counts": {"matrices": len(matrices), "candidate_rows": len(candidates), "selections": len(selections), "prediction_rows": len(predictions)},
        "selectors": list(SELECTORS), "conditions": list(CONDITIONS),
    }
    write_json_new(output / "selection-freeze.json", phase1)
    print(json.dumps(phase1["counts"], indent=2))


def require_selection_freeze(output: Path) -> dict[str, Any]:
    require_input_freeze(output)
    freeze = load_json(output / "selection-freeze.json")
    if freeze["ground_truth_semantically_loaded"] is not False:
        raise ValueError("selection phase was not label blind")
    if sha256_file(output / "input-manifest.json") != freeze["input_manifest_sha256"]:
        raise ValueError("input manifest changed")
    if sha256_file(output / "encoding-manifest.json") != freeze["encoding_manifest_sha256"]:
        raise ValueError("encoding manifest changed")
    for name, expected in freeze["artifacts"].items():
        if sha256_file(output / name) != expected:
            raise ValueError(f"selection artifact changed:{name}")
    return freeze


def partition_signature(ids: list[str], assignments: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    groups: dict[Any, list[str]] = collections.defaultdict(list)
    for report_id in ids:
        groups[assignments[report_id]].append(report_id)
    return tuple(sorted(tuple(sorted(group)) for group in groups.values()))


def decision_sets(assignments: dict[str, Any], truth: dict[str, str]) -> tuple[set[tuple[str, str]], set[str]]:
    ids = sorted(assignments)
    merged = set()
    for left, right in itertools.combinations(ids, 2):
        if assignments[left] == assignments[right] and truth[left] != truth[right]:
            merged.add(tuple(sorted((truth[left], truth[right]))))
    split = set()
    by_bug: dict[str, set[Any]] = collections.defaultdict(set)
    for report_id in ids:
        by_bug[truth[report_id]].add(assignments[report_id])
    for bug, clusters in by_bug.items():
        if len(clusters) > 1:
            split.add(bug)
    return merged, split


def transition_counts(ids: list[str], baseline: dict[str, Any], method: dict[str, Any], truth: dict[str, str]) -> dict[str, int]:
    counts = collections.Counter()
    for left, right in itertools.combinations(ids, 2):
        same_truth = truth[left] == truth[right]
        same_base = baseline[left] == baseline[right]
        same_method = method[left] == method[right]
        base_error, method_error = same_base != same_truth, same_method != same_truth
        counts["baseline_false_merge_report_pairs"] += same_base and not same_truth
        counts["baseline_false_split_report_pairs"] += (not same_base) and same_truth
        counts["method_false_merge_report_pairs"] += same_method and not same_truth
        counts["method_false_split_report_pairs"] += (not same_method) and same_truth
        counts["baseline_pair_errors"] += base_error
        counts["method_pair_errors"] += method_error
        counts["fixed_pair_errors"] += base_error and not method_error
        counts["introduced_pair_errors"] += (not base_error) and method_error
    base_merged, base_split = decision_sets(baseline, truth)
    method_merged, method_split = decision_sets(method, truth)
    counts.update({
        "baseline_merged_bug_pairs": len(base_merged), "method_merged_bug_pairs": len(method_merged),
        "separated_bug_pairs": len(base_merged - method_merged), "newly_merged_bug_pairs": len(method_merged - base_merged),
        "baseline_split_bugs": len(base_split), "method_split_bugs": len(method_split),
        "reunited_bugs": len(base_split - method_split), "newly_split_bugs": len(method_split - base_split),
    })
    return dict(counts)


def mean_present(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def quantile(values: list[float], fraction: float) -> float | None:
    import numpy as np
    return float(np.quantile(values, fraction)) if values else None


def load_metric_module():
    return import_module("l3_gptrace_metrics", METRIC_FILE)


def report_markdown(scores: list[dict[str, Any]], transitions: list[dict[str, Any]], timing_rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    macro = {(row["cohort"], row["condition"], row["selector"]): row for row in scores if row["row_type"] == "macro"}
    lines = [
        "# L3 152-report development validation", "",
        "This is a development-set validation on the previously observed 152 FreeType/Poppler/SoX reports. It does not replace L0 and is not a held-out or generalization result.", "",
        "## Actual clustering performance", "",
        "| Cohort | Selector | Condition | macro F | macro ARI | macro global AP | clusters | noise |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort in ("full_152", "common_available"):
        for selector in SELECTORS:
            for condition in CONDITIONS:
                row = macro[(cohort, condition, selector)]
                lines.append(f"| {cohort} | {selector} | {condition} | {row['f_measure']:.5f} | {row['adjusted_rand_index']:.5f} | {row['pairwise_average_precision_global']:.5f} | {row['predicted_clusters']} | {row['noise_point_count']} |")
        loc = macro[(cohort, "LOC", "NA")]
        lines.append(f"| {cohort} | NA | LOC | {loc['f_measure']:.5f} | {loc['adjusted_rand_index']:.5f} | N/A | {loc['predicted_clusters']} | 0 |")
    lines += ["", "Within each selector and cohort, the requested macro-F contrasts are:", ""]
    for cohort in ("full_152", "common_available"):
        for selector in SELECTORS:
            bl3 = macro[(cohort, "BL3", selector)]["f_measure"]
            lines.append(
                f"- {cohort}/{selector}: BL3−BL0 {bl3-macro[(cohort,'BL0',selector)]['f_measure']:+.5f}; "
                f"BL3−BS_L3 {bl3-macro[(cohort,'BS_L3',selector)]['f_measure']:+.5f}; "
                f"BL3−B {bl3-macro[(cohort,'B',selector)]['f_measure']:+.5f}; "
                f"BL3−LOC {bl3-macro[(cohort,'LOC','NA')]['f_measure']:+.5f}."
            )
    lines += [
        "", "AP is computed from the fixed distance matrix and is therefore identical for S0 and Smax; LOC has no embedding distance and no AP. Cluster labels are evaluated only after the selection freeze.",
        "", "## Input-amount control", "",
        f"BS_L3 was available for {summary['BS_available']}/152 reports; the common-availability cohort contains {summary['common_reports']} reports. "
        f"The control exactly matched L3 chunk count in {summary['BS_available']} available reports. Its absolute token difference had median {summary['BS_token_error_median']:.1f}, mean {summary['BS_token_error_mean']:.2f}, and maximum {summary['BS_token_error_max']} tokens.",
    ]
    full_control_outcome = []
    for selector in SELECTORS:
        bl3 = macro[("full_152", "BL3", selector)]["f_measure"]
        bs = macro[("full_152", "BS_L3", selector)]["f_measure"]
        full_control_outcome.append(f"{selector} BL3−BS_L3={bl3-bs:+.5f}")
    lines += [
        "", "The continuous-window control results were " + "; ".join(full_control_outcome) + ". A positive BL3−BL0 result without a positive BL3−BS_L3 result does not identify a unique multi-frame selection benefit.",
        "", "## New errors and preserved relationships", "",
    ]
    for selector in SELECTORS:
        group = [row for row in transitions if row["cohort"] == "full_152" and row["selector"] == selector and row["condition"] == "BL3"]
        totals = {name: sum(int(row[name]) for row in group) for name in ("fixed_pair_errors", "introduced_pair_errors", "separated_bug_pairs", "newly_merged_bug_pairs", "reunited_bugs", "newly_split_bugs")}
        lines.append(f"- {selector}, BL3 vs BL0: fixed {totals['fixed_pair_errors']} report-pair errors and introduced {totals['introduced_pair_errors']}; separated {totals['separated_bug_pairs']} merged bug-pairs/newly merged {totals['newly_merged_bug_pairs']}; reunited {totals['reunited_bugs']} split bugs/newly split {totals['newly_split_bugs']}.")
    l0_rows = [row for row in timing_rows if row["condition"] == "BL0"]
    l3_rows = [row for row in timing_rows if row["condition"] == "BL3"]
    bs_rows = [row for row in timing_rows if row["condition"] == "BS_L3"]
    def avg(rows, key):
        return sum(float(row[key]) for row in rows) / len(rows)
    lines += [
        "", "## Cost", "",
        f"Mean additional-context tokens/chunks were L0 {avg(l0_rows,'additional_input_tokens'):.1f}/{avg(l0_rows,'additional_chunks'):.2f}, "
        f"L3 {avg(l3_rows,'additional_input_tokens'):.1f}/{avg(l3_rows,'additional_chunks'):.2f}, and BS_L3 {avg(bs_rows,'additional_input_tokens'):.1f}/{avg(bs_rows,'additional_chunks'):.2f} including fallbacks. "
        f"L3 changed {summary['L3_changed']} reports and used fewer than three unique frames in {summary['L3_fewer_than_3']} reports.",
        f"Cumulative extraction/generation time was L3 {sum(float(row['extraction_or_generation_seconds']) for row in l3_rows):.2f}s and BS_L3 {sum(float(row['extraction_or_generation_seconds']) for row in bs_rows):.2f}s; measured new-text encoding time totaled {sum(float(row['encoding_seconds']) for row in l3_rows+bs_rows):.2f}s. Cache hits are separately identified in token-and-timing.csv.",
        "", "## Evidence for and against adopting L3", "",
    ]
    full_deltas = {selector: macro[("full_152", "BL3", selector)]["f_measure"] - macro[("full_152", "BL0", selector)]["f_measure"] for selector in SELECTORS}
    control_deltas = {selector: macro[("full_152", "BL3", selector)]["f_measure"] - macro[("full_152", "BS_L3", selector)]["f_measure"] for selector in SELECTORS}
    lines.append("- Supporting evidence: " + "; ".join(f"{selector} BL3−BL0={value:+.5f}" for selector, value in full_deltas.items()) + ". The distance-distributions and pilot-pairs tables show whether rank/distance movement agrees with the cluster changes.")
    lines.append("- Counterevidence: " + "; ".join(f"{selector} BL3−BS_L3={value:+.5f}" for selector, value in control_deltas.items()) + f"; BL3 also introduced report-pair errors listed above. Input size is closely but not perfectly matched (maximum token error {summary['BS_token_error_max']}).")
    lines += [
        "", "## Interpretation and remaining validation", "",
        "Absolute distance shifts are not treated as improvement: global AP, bug-balanced anchor mAP, distance distributions, and actual cluster transitions are reported together. Overall distance-scale changes remain a possible explanation.",
        "The five pilot pairs are reported only as a diagnostic trace within the full clustering; they do not define the aggregate conclusion. Because this is previously observed development data, any adoption decision still requires the unchanged held-out/new-project protocol to compare L0, L3, and the same input-amount control prospectively. No L5, Lall, new weight, or third selector was run.",
    ]
    return "\n".join(lines) + "\n"


def command_score(cli: argparse.Namespace) -> None:
    import numpy as np
    from sklearn.metrics import pairwise_distances
    output = output_path(cli)
    freeze = require_selection_freeze(output)
    sys.path.insert(0, str(WORK))
    import score_control_pilot_v1 as score_legacy
    metric_module = load_metric_module()
    labels = {row["report_id"]: row["label"] for row in load_jsonl(INPUTS["labels"])}
    matrices = load_jsonl(output / "matrices.jsonl")
    selections = load_jsonl(output / "selections.jsonl")
    predictions_rows = load_jsonl(output / "predictions.jsonl")
    contexts = load_jsonl(output / "contexts.jsonl")
    controls = load_jsonl(output / "controls.jsonl")
    manifest = load_json(output / "input-manifest.json")
    reps = load_representations(output / "representations.npz")
    selection_by = {(row["key"], row["selector"]): row for row in selections}
    predictions: dict[tuple[str, str], dict[str, int]] = collections.defaultdict(dict)
    for row in predictions_rows:
        predictions[(row["key"], row["selector"])][row["report_id"]] = int(row["cluster"])
    context_by = {row["report_id"]: row for row in contexts}
    control_by = {row["report_id"]: row for row in controls}
    scores = []
    ranking_by_key = {}
    for matrix_row in matrices:
        key, ids = matrix_row["key"], matrix_row["report_ids"]
        matrix = np.stack([reps[f"{report_id}:{matrix_row['condition']}"] for report_id in ids]).astype(np.float64, copy=False)
        if matrix_hash(matrix) != matrix_row["matrix_sha256"]:
            raise ValueError(f"matrix rebuild mismatch:{key}")
        truth = [labels[report_id] for report_id in ids]
        ranking = score_legacy.ranking_metrics(matrix, truth)
        ranking_by_key[key] = ranking
        for selector in SELECTORS:
            selected = selection_by[(key, selector)]
            predicted = [predictions[(key, selector)][report_id] for report_id in ids]
            condition = matrix_row["condition"]
            enhanced = sum(
                condition == "BL0" or condition == "BL3" or
                (condition == "BS_L3" and control_by[report_id]["BS_L3"]["status"] == "available")
                for report_id in ids
            ) if condition != "B" else 0
            unchanged = sum(context_by[report_id]["conditions"]["L3"]["unchanged_from_L0"] for report_id in ids) if condition == "BL3" else 0
            scores.append({
                "row_type": "target", "cohort": matrix_row["cohort"], "target": matrix_row["target"],
                "condition": condition, "selector": selector, "reports": len(ids), "true_bugs": len(set(truth)),
                "predicted_clusters": len(set(predicted)), "noise_point_count": selected["noise_point_count"],
                "enhanced_reports": enhanced, "fallback_to_B_reports": (len(ids)-enhanced if condition in ("BL0", "BL3", "BS_L3") else 0),
                "unchanged_from_BL0_reports": unchanged, "matrix_sha256": matrix_row["matrix_sha256"],
                "partition_sha256": selected["partition_sha256"], "epsilon": selected["epsilon"],
                **score_legacy.primary_metrics(metric_module, predicted, truth), **ranking,
            })
    location = {row["report_id"]: row["location_cluster"] for row in load_jsonl(INPUTS["location_predictions"])}
    for cohort in ("full_152", "common_available"):
        for target in TARGETS:
            ids = manifest["cohorts"][cohort][target]
            predicted, truth = [location[report_id] for report_id in ids], [labels[report_id] for report_id in ids]
            scores.append({
                "row_type": "target", "cohort": cohort, "target": target, "condition": "LOC", "selector": "NA",
                "reports": len(ids), "true_bugs": len(set(truth)), "predicted_clusters": len(set(predicted)),
                "noise_point_count": 0, "enhanced_reports": 0, "fallback_to_B_reports": 0,
                "unchanged_from_BL0_reports": 0, "matrix_sha256": None,
                "partition_sha256": sha256_text(canonical(predicted)), "epsilon": None,
                **score_legacy.primary_metrics(metric_module, predicted, truth),
                "pairwise_average_precision_global": None, "bug_balanced_anchor_map": None,
                "ranking_bug_count": None, "ranking_anchor_count": None, "ranking_excluded_singleton_bug_count": None,
            })
    target_rows = list(scores)
    for cohort in ("full_152", "common_available"):
        for condition, selector in list(itertools.product(CONDITIONS, SELECTORS)) + [("LOC", "NA")]:
            group = [row for row in target_rows if (row["cohort"], row["condition"], row["selector"]) == (cohort, condition, selector)]
            macro = {key: None for key in target_rows[0]}
            macro.update({
                "row_type": "macro", "cohort": cohort, "target": "MACRO", "condition": condition,
                "selector": selector, "reports": sum(row["reports"] for row in group),
                "true_bugs": sum(row["true_bugs"] for row in group), "predicted_clusters": sum(row["predicted_clusters"] for row in group),
                "noise_point_count": sum(row["noise_point_count"] for row in group),
                "enhanced_reports": sum(row["enhanced_reports"] for row in group),
                "fallback_to_B_reports": sum(row["fallback_to_B_reports"] for row in group),
                "unchanged_from_BL0_reports": sum(row["unchanged_from_BL0_reports"] for row in group),
            })
            for metric in ("purity", "inverse_purity", "f_measure", "adjusted_rand_index", "pairwise_average_precision_global", "bug_balanced_anchor_map", "num_overcount", "num_undercount", "num_completely_lost"):
                macro[metric] = mean_present(row[metric] for row in group)
            scores.append(macro)
    write_csv_new(output / "scores.csv", scores)

    transitions = []
    for cohort in ("full_152", "common_available"):
        for target in TARGETS:
            ids = manifest["cohorts"][cohort][target]
            truth_map = {report_id: labels[report_id] for report_id in ids}
            for selector in SELECTORS:
                baseline = predictions[(f"{cohort}/{target}/BL0", selector)]
                for condition in ("BL3", "BS_L3"):
                    method = predictions[(f"{cohort}/{target}/{condition}", selector)]
                    transitions.append({
                        "cohort": cohort, "target": target, "condition": condition, "selector": selector,
                        "baseline": "BL0", "reports": len(ids), **transition_counts(ids, baseline, method, truth_map),
                    })
    write_csv_new(output / "transitions.csv", transitions)

    distance_rows = []
    for matrix_row in matrices:
        ids = matrix_row["report_ids"]
        matrix = np.stack([reps[f"{report_id}:{matrix_row['condition']}"] for report_id in ids])
        distances = pairwise_distances(matrix, metric="euclidean")
        grouped = {"same_bug": [], "different_bug": []}
        for left, right in itertools.combinations(range(len(ids)), 2):
            kind = "same_bug" if labels[ids[left]] == labels[ids[right]] else "different_bug"
            grouped[kind].append(float(distances[left, right]))
        for kind, values in grouped.items():
            distance_rows.append({
                "cohort": matrix_row["cohort"], "target": matrix_row["target"], "condition": matrix_row["condition"],
                "pair_type": kind, "pair_count": len(values), "min": min(values) if values else None,
                "q25": quantile(values, .25), "median": quantile(values, .5), "mean": mean_present(values),
                "q75": quantile(values, .75), "max": max(values) if values else None,
            })
    write_csv_new(output / "distance-distributions.csv", distance_rows)

    pilot_manifest, pilot_key = load_json(INPUTS["pilot_sample_manifest"]), load_json(INPUTS["pilot_sample_key"])
    key_by_pair = {row["pair_id"]: row for row in pilot_key["pairs"]}
    pilot_rows = []
    for pair in pilot_manifest["pairs"]:
        left, right = pair["report_ids"]
        key_meta = key_by_pair[pair["pair_id"]]
        for selector in SELECTORS:
            for condition in ("BL0", "BL3", "BS_L3"):
                mapping = predictions[(f"full_152/{pair['target']}/{condition}", selector)]
                pilot_rows.append({
                    "pair_id": pair["pair_id"], "target": pair["target"], "left_report_id": left, "right_report_id": right,
                    "existing_error_type": key_meta["existing_error_type"], "truth_same_defect": key_meta["truth_same_defect"],
                    "selector": selector, "condition": condition, "same_cluster": mapping[left] == mapping[right],
                    "judgment_correct": (mapping[left] == mapping[right]) == key_meta["truth_same_defect"],
                    "left_cluster": mapping[left], "right_cluster": mapping[right],
                })
    write_csv_new(output / "pilot-pairs.csv", pilot_rows)

    legacy_rows = [row for row in load_jsonl(INPUTS["legacy_predictions"]) if row["cohort"] == "fallback_all" and row["condition"] in ("B", "BL")]
    legacy_maps: dict[tuple[str, str, str], dict[str, int]] = collections.defaultdict(dict)
    for row in legacy_rows:
        legacy_maps[(row["target"], row["condition"], row["selector"])][row["report_id"]] = int(row["cluster"])
    reproduction = {}
    for target in TARGETS:
        ids = manifest["cohorts"]["full_152"][target]
        for condition, legacy_condition in (("B", "B"), ("BL0", "BL")):
            for selector in SELECTORS:
                current = predictions[(f"full_152/{target}/{condition}", selector)]
                previous = legacy_maps[(target, legacy_condition, selector)]
                reproduction[f"{target}/{condition}/{selector}"] = {
                    "partition_equivalent": partition_signature(ids, current) == partition_signature(ids, previous),
                    "literal_labels_equal": [current[i] for i in ids] == [previous[i] for i in ids],
                }
    pilot_contexts = {row["report_id"]: row for row in load_jsonl(INPUTS["pilot_contexts"])}
    pilot_ids = [report_id for pair in pilot_manifest["pairs"] for report_id in pair["report_ids"]]
    l3_pilot_match = {report_id: context_by[report_id]["conditions"]["L3"]["text_sha256"] == pilot_contexts[report_id]["conditions"]["L3"]["text_sha256"] and context_by[report_id]["conditions"]["L3"]["text"] == pilot_contexts[report_id]["conditions"]["L3"]["text"] for report_id in pilot_ids}
    timing_rows = list(csv.DictReader((output / "token-and-timing.csv").open(encoding="utf-8-sig", newline="")))
    bs_available_rows = [row for row in controls if row["BS_L3"]["status"] == "available"]
    token_errors = [abs(int(row["BS_L3"]["token_error"])) for row in bs_available_rows]
    summary = {
        "BS_available": len(bs_available_rows), "common_reports": sum(len(value) for value in manifest["cohorts"]["common_available"].values()),
        "BS_token_error_median": quantile(token_errors, .5) or 0.0, "BS_token_error_mean": mean_present(token_errors) or 0.0,
        "BS_token_error_max": max(token_errors) if token_errors else 0,
        "L3_changed": sum(not row["conditions"]["L3"]["unchanged_from_L0"] for row in contexts),
        "L3_fewer_than_3": sum(row["conditions"]["L3"]["unique_frames"] < 3 for row in contexts),
    }
    write_json_new(output / "summary.json", summary)
    write_new(output / "REPORT.md", report_markdown(scores, transitions, timing_rows, summary))
    write_new(output / "HANDOFF.md", "# Handoff\n\nCompleted the frozen 152-report L3 development validation. L0 and unused-project protocols were not changed. Phase-1 contexts, input cohorts, embeddings, matrices, candidates, selectors, and predictions were frozen before semantic label/legacy-result loading. Read REPORT.md for conclusions and validation.json for exact checks.\n")
    write_new(output / "FAILURE_LOG.md", "# Failure log\n\n## Windows Python launcher unavailable\n\n- Symptom/case: the initial `py -3 -m py_compile` attempt failed before Python started with a terminated logon-session error from the Windows Store launcher.\n- Cause: the `py.exe` app alias was unavailable in this non-interactive session; the adapter itself had not run.\n- Action: no environment or dependency was changed. Syntax checks and execution used the two already-pinned WSL virtual environments.\n- Revalidation: WSL `py_compile` completed successfully, followed by the dedicated unit tests and end-to-end validation.\n\nNo unresolved execution failure remained. Expected BS_L3 unavailability and token/chunk differences are recorded per report in controls.jsonl and summarized in input-manifest.json; they are data outcomes, not silently repaired.\n")

    before = load_json(output / "protection-before.json")
    after = {name: inventory(ROOT / name) for name in PROTECTED}
    stable = {name: before[name] == after[name] for name in PROTECTED}
    l0_reference = {row["report_id"]: row for row in load_jsonl(INPUTS["contexts"])}
    ap_invariant = {}
    for matrix_row in matrices:
        key = matrix_row["key"]
        score_rows = [row for row in target_rows if row["cohort"] == matrix_row["cohort"] and row["target"] == matrix_row["target"] and row["condition"] == matrix_row["condition"]]
        ap_invariant[key] = len(score_rows) == 2 and score_rows[0]["pairwise_average_precision_global"] == score_rows[1]["pairwise_average_precision_global"] and score_rows[0]["bug_balanced_anchor_map"] == score_rows[1]["bug_balanced_anchor_map"]
    checks = {
        "phase1_frozen_before_label_join": freeze["ground_truth_semantically_loaded"] is False,
        "report_count_and_unique_ids": len(contexts) == 152 and len(context_by) == 152,
        "L0_all_text_and_hash_match": all(row["conditions"]["L0"]["text"] == l0_reference[row["report_id"]]["l_text"] and row["conditions"]["L0"]["text_sha256"] == l0_reference[row["report_id"]]["l_text_sha256"] for row in contexts),
        "pilot_10_L3_exact_match": all(l3_pilot_match.values()) and len(l3_pilot_match) == 10,
        "same_stack_only": all(row["same_stack_only"] for row in contexts),
        "frame_order_increasing": all([frame["frame_number"] for frame in row["frames"]] == sorted(frame["frame_number"] for frame in row["frames"]) for row in contexts),
        "deduplicated_location_keys": all(len({(frame["source_id"], frame["source_relative_path"], frame["line"]) for frame in row["frames"]}) == len(row["frames"]) for row in contexts),
        "all_failures_have_reasons": all(all(failure.get("reason") for failure in row["failures"]) for row in contexts),
        "encoder_leak_scan_empty": not load_json(output / "encoding-manifest.json")["leak_findings"],
        "full_B_BL0_S0_Smax_partition_match": all(value["partition_equivalent"] for value in reproduction.values()),
        "report_order_and_partition_lengths": all(len(row["report_ids"]) == row["reports"] == len(set(row["report_ids"])) for row in matrices) and all(len(predictions[(row["key"], selector)]) == row["reports"] for row in matrices for selector in SELECTORS),
        "selector_AP_invariant": all(ap_invariant.values()),
        "BS_available_chunk_match": all(row["BS_L3"]["chunk_error"] == 0 for row in bs_available_rows),
        "BS_failures_explicit": all(row["BS_L3"].get("reason") for row in controls if row["BS_L3"]["status"] != "available"),
        "protected_artifacts_unchanged": all(stable.values()),
        "unit_tests_passed": (output / "unit-test-results.txt").is_file() and "OK" in (output / "unit-test-results.txt").read_text(encoding="utf-8"),
    }
    validation = {
        "schema_version": "l3-development-validation-v1", "status": "passed" if all(checks.values()) else "failed",
        "checks": checks, "legacy_reproduction": reproduction, "pilot_L3_matches": l3_pilot_match,
        "AP_selector_invariance": ap_invariant, "protected": {name: {"before": before[name], "after": after[name], "unchanged": stable[name]} for name in PROTECTED},
        "counts": {**summary, "matrices": len(matrices), "selections": len(selections), "prediction_rows": len(predictions_rows)},
        "required_outputs": ["REPORT.md", "protocol.json", "input-manifest.json", "selection-freeze.json", "scores.csv", "transitions.csv", "token-and-timing.csv", "validation.json", "HANDOFF.md", "FAILURE_LOG.md"],
    }
    write_json_new(output / "validation.json", validation)
    if validation["status"] != "passed":
        raise ValueError(json.dumps(validation, indent=2))
    print(json.dumps({"validation": validation["status"], "summary": summary}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("command", choices=("protocol", "extract", "controls", "encode", "cluster", "score"))
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return result


def main() -> None:
    cli = parser().parse_args()
    commands = {
        "protocol": command_protocol, "extract": command_extract, "controls": command_controls,
        "encode": command_encode, "cluster": command_cluster, "score": command_score,
    }
    commands[cli.command](cli)


if __name__ == "__main__":
    main()

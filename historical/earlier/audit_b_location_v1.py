"""Read-only audit of source-coordinate loss in B encoder preprocessing.

This adapter never writes a shared input, cache, embedding, or clustering
artifact.  It compares the saved raw report/trace files, the pinned GPTrace
preprocessing replay, the current additional redaction replay, and saved
encoder inputs.  A replay is explicitly not treated as proof of a historical
implementation unless the relevant manifest records its code hash.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
REVISED = ROOT / "outputs/revised"
NEW = ROOT / "outputs/new-project-evaluation-v2"
OUTPUT_DEFAULT = ROOT / "outputs/b-location-audit-v1"
ARTIFACT_ROOT = Path("/RESEARCH_HOME/gptrace-artifacts")
DATA = ARTIFACT_ROOT / "reproduce/evaluation/data_sources"
GP_PREPROCESSING = ARTIFACT_ROOT / "gptrace/src/gptrace/preprocessing.py"
SOURCE_SUFFIX = r"(?:c|cc|cpp|cxx|h|hh|hpp)"
ABS_POSIX_SOURCE = re.compile(
    rf"(?<![A-Za-z0-9_.-])(?P<path>/(?:[^\s()\"'<>:]+/)*[^\s()\"'<>:]+\.{SOURCE_SUFFIX}):(?P<line>\d+)(?::(?P<column>\d+))?"
)
WINDOWS_SOURCE = re.compile(
    rf"(?P<path>[A-Za-z]:[\\/](?:[^\s()\"'<>:]+[\\/])*[^\s()\"'<>:]+\.{SOURCE_SUFFIX}):(?P<line>\d+)(?::(?P<column>\d+))?"
)
RELATIVE_SOURCE = re.compile(
    rf"(?<![A-Za-z0-9_/\\])(?P<path>(?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_.-]+\.{SOURCE_SUFFIX}):(?P<line>\d+)(?::(?P<column>\d+))?"
)
POSIX_PATH = re.compile(r"/(?:[^\s\"'<>(),]+/)*[^\s\"'<>(),]+")
LABEL = re.compile(r"\bpoc_[A-Za-z0-9_-]+\b")
BENCHMARK = re.compile(r"\bMAGMA_[A-Z0-9_]+\b|\bPDF\d{3}\b")
SOURCE_LIKE_UNPARSED = re.compile(r"\.[A-Za-z]{1,4}:\d+")
TEXT_KINDS = ("trace", "no_args_trace", "asan")
DEV_TARGETS = ("freetype__char2svg", "poppler__pdfimages", "soxmp3__sox")


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


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def write_csv_new(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows and not fields:
        raise ValueError("CSV needs rows or fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_new(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def import_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalized_path(value: str) -> str:
    return value.replace("\\", "/")


def scan_source_locations(text: str) -> list[dict[str, Any]]:
    """Identify source-coordinate tokens, retaining path/line/column correspondence."""
    found: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for kind, pattern in (("absolute_posix", ABS_POSIX_SOURCE), ("windows", WINDOWS_SOURCE), ("relative_or_basename", RELATIVE_SOURCE)):
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(not (end <= left or start >= right) for left, right in occupied):
                continue
            path = match.group("path")
            found.append({
                "path": path, "path_normalized": normalized_path(path), "basename": normalized_path(path).rsplit("/", 1)[-1],
                "line": int(match.group("line")), "column": int(match.group("column")) if match.group("column") else None,
                "kind": kind, "span": (start, end), "token": match.group(0),
            })
            occupied.append((start, end))
    return sorted(found, key=lambda item: item["span"])


def location_counter(locations: list[dict[str, Any]], field: str = "full") -> collections.Counter:
    if field == "full":
        values = [(item["path_normalized"], item["line"], item["column"]) for item in locations]
    elif field == "filename":
        values = [item["path_normalized"] for item in locations]
    elif field == "line":
        values = [(item["path_normalized"], item["line"]) for item in locations]
    elif field == "column":
        values = [(item["path_normalized"], item["line"], item["column"]) for item in locations if item["column"] is not None]
    else:
        raise ValueError(f"unknown location field:{field}")
    return collections.Counter(values)


def field_status(before: list[dict[str, Any]], after: list[dict[str, Any]], field: str) -> str:
    initial = location_counter(before, field)
    if not initial:
        return "not_observed"
    retained = initial & location_counter(after, field)
    if retained == initial:
        return "preserved"
    if not retained:
        return "removed"
    return "partially_preserved"


def source_like_unparsed(text: str, locations: list[dict[str, Any]]) -> bool:
    return bool(SOURCE_LIKE_UNPARSED.search(text)) and not locations


def source_outcome(raw: list[dict[str, Any]], gp: list[dict[str, Any]], redacted: list[dict[str, Any]], raw_text: str) -> str:
    raw_count = location_counter(raw)
    if not raw_count:
        return "parse_uncertain" if source_like_unparsed(raw_text, raw) else "no_original_source_coordinate_observed"
    gp_count = location_counter(gp)
    final_count = location_counter(redacted)
    gp_retained, final_retained = raw_count & gp_count, raw_count & final_count
    if not gp_retained:
        return "removed_or_changed_at_gptrace_preprocessing"
    if gp_retained != raw_count:
        if final_retained:
            return "mixed_gptrace_and_redaction_or_partial_loss"
        return "gptrace_preprocessing_partial_loss_then_redaction_loss"
    if final_retained == raw_count:
        return "preserved_to_stored_text"
    if not final_retained:
        return "removed_at_additional_redaction"
    return "partially_removed_at_additional_redaction"


def count_runtime_paths(text: str, locations: list[dict[str, Any]]) -> int:
    source_spans = [item["span"] for item in locations]
    count = 0
    for match in POSIX_PATH.finditer(text):
        start, end = match.span()
        if any(not (end <= left or start >= right) for left, right in source_spans):
            continue
        count += 1
    return count


def asan_location_breakdown(text: str) -> dict[str, int]:
    first_block = text.split("\n\n", 1)[0]
    stack = "\n".join(line for line in first_block.splitlines() if re.match(r"^\s*#\d+\b", line))
    summary = "\n".join(line for line in text.splitlines() if line.lstrip().startswith("SUMMARY:"))
    return {
        "initial_stack_source_locations": len(scan_source_locations(stack)),
        "summary_source_locations": len(scan_source_locations(summary)),
        "all_source_locations": len(scan_source_locations(text)),
    }


def current_stage_kind(text_kind: str, raw: str, gp: str) -> str:
    if raw == gp:
        return "none"
    if text_kind == "asan":
        before, after = asan_location_breakdown(raw), asan_location_breakdown(gp)
        if before["initial_stack_source_locations"] > after["initial_stack_source_locations"]:
            return "gptrace_remove_asan_traces_initial_stack"
        return "gptrace_asan_other_transformation_or_uncertain"
    return "gptrace_duplicate_or_argument_transformation_or_uncertain"


def boolean_count(text: str, marker: str | re.Pattern[str]) -> bool:
    if isinstance(marker, str):
        return marker in text
    return bool(marker.search(text))


def source_descriptor(locations: list[dict[str, Any]]) -> str | None:
    if not locations:
        return None
    first = locations[0]
    column = "" if first["column"] is None else f":{first['column']}"
    return f"{first['basename']}:{first['line']}{column}"


def safe_descriptor(text: str, report_name: str, locations: list[dict[str, Any]]) -> dict[str, Any]:
    """Return only categories and a basename coordinate, never a raw path or label."""
    return {
        "source_coordinate": source_descriptor(locations),
        "runtime_path_count": count_runtime_paths(text, locations),
        "input_filename_present": report_name in text,
        "input_placeholder_present": "<INPUT_FILE>" in text,
        "label_marker_present": bool(LABEL.search(text)),
        "label_placeholder_present": "<LABEL_PATH>" in text,
        "path_placeholder_count": text.count("<PATH>"),
        "benchmark_marker_present": bool(BENCHMARK.search(text)),
    }


def make_args(argparse_module: Any, trace_path: Path, report_path: Path):
    return argparse_module.Namespace(
        remove_arguments=False, remove_arguments_extra=True, remove_duplicate_frames=True,
        remove_asan_traces=True, input_dir=trace_path.parents[1], asan_dir=report_path.parents[1],
    )


def input_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join manifests to saved inputs without persisting raw label-bearing paths."""
    dev_contexts = {row["report_id"]: row for row in load_jsonl(REVISED / "contexts-v4.jsonl")}
    dev_primary = {row["report_id"]: row for row in load_jsonl(REVISED / "encoder-inputs-v4.jsonl")}
    dev_control = {row["report_id"]: row for row in load_jsonl(REVISED / "control-eval-encoder-inputs-v1.jsonl")}
    dev_manifest = load_json(REVISED / "encoder-input-v4-manifest.json")
    if set(dev_contexts) != set(dev_primary) or set(dev_primary) != set(dev_control):
        raise ValueError("development_report_id_set_mismatch")
    records = []
    for report_id, context in dev_contexts.items():
        relative_report = context["report_relative_path"]
        target = context["target"]
        report = DATA / relative_report
        relative = report.relative_to(DATA / target / "asan_logs")
        trace = DATA / target / "traces" / relative
        records.append({
            "corpus": "development_v4", "report_id": report_id, "target": target,
            "report": report, "trace": trace, "report_name": report.name,
            "stored_primary": dev_primary[report_id]["texts"], "stored_secondary": dev_control[report_id]["texts"],
            "historical_redaction_evidence": "prepare_encoder_inputs_hash_not_recorded_in_v4_manifest",
        })
    new_manifest = load_json(NEW / "input-manifest.json")
    new_primary = {row["report_id"]: row for row in load_jsonl(NEW / "baseline-inputs.jsonl")}
    new_joined = {row["report_id"]: row for row in load_jsonl(NEW / "encoder-inputs.jsonl")}
    samples = {row["report_id"]: row for row in new_manifest["samples"]}
    if set(samples) != set(new_primary) or set(samples) != set(new_joined):
        raise ValueError("new_project_report_id_set_mismatch")
    for report_id, sample in samples.items():
        report, trace = DATA / sample["report_relative_path"], DATA / sample["trace_relative_path"]
        records.append({
            "corpus": "new_project_v2", "report_id": report_id, "target": sample["target"],
            "report": report, "trace": trace, "report_name": report.name,
            "stored_primary": new_primary[report_id]["texts"], "stored_secondary": new_joined[report_id]["texts"],
            "historical_redaction_evidence": "prepare_encoder_inputs_hash_recorded_in_new_extraction_manifest",
        })
    records.sort(key=lambda row: (row["corpus"], row["target"], row["report_id"]))
    metadata = {"development_manifest": dev_manifest, "new_manifest": new_manifest}
    return records, metadata


def source_input_inventory(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    rows = []
    for row in records:
        if not row["report"].is_file() or not row["trace"].is_file():
            raise FileNotFoundError(f"missing saved input:{row['corpus']}:{row['report_id']}")
        rows.append({
            "corpus": row["corpus"], "report_id": row["report_id"], "target": row["target"],
            "report_sha256": sha256_file(row["report"]), "trace_sha256": sha256_file(row["trace"]),
        })
    return rows, sha256_text(canonical(rows))


def audit_record(record: dict[str, Any], pre: Any, baseline: Any, argparse_module: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    args = make_args(argparse_module, record["trace"], record["report"])
    processed = pre.read_trace_and_asan(args, record["trace"])
    raw_trace = record["trace"].read_text(encoding="utf-8", errors="replace")
    raw_asan = record["report"].read_text(encoding="utf-8", errors="replace")
    raw_texts = {"trace": raw_trace, "no_args_trace": raw_trace, "asan": raw_asan}
    result, private = [], []
    for text_kind in TEXT_KINDS:
        raw, gp = raw_texts[text_kind], processed[text_kind]
        redacted = baseline.redact(gp, record["report_name"])
        saved_primary, saved_secondary = record["stored_primary"].get(text_kind), record["stored_secondary"].get(text_kind)
        raw_locations, gp_locations, redacted_locations = scan_source_locations(raw), scan_source_locations(gp), scan_source_locations(redacted)
        stored_locations = scan_source_locations(saved_primary or "")
        raw_breakdown = asan_location_breakdown(raw) if text_kind == "asan" else None
        gp_breakdown = asan_location_breakdown(gp) if text_kind == "asan" else None
        redacted_breakdown = asan_location_breakdown(redacted) if text_kind == "asan" else None
        raw_counter, gp_counter, redacted_counter = location_counter(raw_locations), location_counter(gp_locations), location_counter(redacted_locations)
        row = {
            "corpus": record["corpus"], "report_id": record["report_id"], "target": record["target"], "text_kind": text_kind,
            "raw_source_coordinate_count": sum(raw_counter.values()), "gp_source_coordinate_count": sum(gp_counter.values()),
            "redacted_source_coordinate_count": sum(redacted_counter.values()), "stored_source_coordinate_count": len(stored_locations),
            "original_source_coordinate_observed": bool(raw_counter),
            "source_coordinate_outcome": source_outcome(raw_locations, gp_locations, redacted_locations, raw),
            "gp_coordinate_occurrences_removed": sum((raw_counter - gp_counter).values()),
            "redaction_coordinate_occurrences_removed": sum((gp_counter - redacted_counter).values()),
            "gp_change_kind": current_stage_kind(text_kind, raw, gp),
            "filename_to_stored": field_status(raw_locations, redacted_locations, "filename"),
            "line_to_stored": field_status(raw_locations, redacted_locations, "line"),
            "column_to_stored": field_status(raw_locations, redacted_locations, "column"),
            "filename_after_gp": field_status(raw_locations, gp_locations, "filename"),
            "line_after_gp": field_status(raw_locations, gp_locations, "line"),
            "column_after_gp": field_status(raw_locations, gp_locations, "column"),
            "source_parse_uncertain": source_like_unparsed(raw, raw_locations),
            "raw_runtime_path_count": count_runtime_paths(raw, raw_locations),
            "gp_runtime_path_count": count_runtime_paths(gp, gp_locations),
            "redacted_runtime_path_count": count_runtime_paths(redacted, redacted_locations),
            "raw_input_filename_present": boolean_count(raw, record["report_name"]),
            "gp_input_filename_present": boolean_count(gp, record["report_name"]),
            "redacted_input_filename_present": boolean_count(redacted, record["report_name"]),
            "redacted_input_placeholder_present": "<INPUT_FILE>" in redacted,
            "raw_label_marker_present": boolean_count(raw, LABEL), "gp_label_marker_present": boolean_count(gp, LABEL),
            "redacted_label_marker_present": boolean_count(redacted, LABEL), "redacted_label_placeholder_present": "<LABEL_PATH>" in redacted,
            "raw_benchmark_marker_present": boolean_count(raw, BENCHMARK), "redacted_benchmark_marker_present": boolean_count(redacted, BENCHMARK),
            "stored_primary_exact_match": redacted == saved_primary,
            "stored_secondary_exact_match": redacted == saved_secondary,
            "raw_sha256": sha256_text(raw), "gp_sha256": sha256_text(gp), "redacted_sha256": sha256_text(redacted),
            "stored_primary_sha256": sha256_text(saved_primary or ""), "stored_secondary_sha256": sha256_text(saved_secondary or ""),
            "historical_redaction_evidence": record["historical_redaction_evidence"],
            "raw_asan_initial_stack_source_coordinate_count": raw_breakdown["initial_stack_source_locations"] if raw_breakdown else None,
            "gp_asan_initial_stack_source_coordinate_count": gp_breakdown["initial_stack_source_locations"] if gp_breakdown else None,
            "redacted_asan_initial_stack_source_coordinate_count": redacted_breakdown["initial_stack_source_locations"] if redacted_breakdown else None,
            "raw_asan_summary_source_coordinate_count": raw_breakdown["summary_source_locations"] if raw_breakdown else None,
            "gp_asan_summary_source_coordinate_count": gp_breakdown["summary_source_locations"] if gp_breakdown else None,
            "redacted_asan_summary_source_coordinate_count": redacted_breakdown["summary_source_locations"] if redacted_breakdown else None,
        }
        result.append(row)
        private.append({
            "row": row, "report_name": record["report_name"], "raw": raw, "gp": gp, "redacted": redacted,
            "raw_locations": raw_locations, "gp_locations": gp_locations, "redacted_locations": redacted_locations,
        })
    return result, private


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["corpus"], row["target"], row["text_kind"])].append(row)
        groups[(row["corpus"], "__all__", row["text_kind"])].append(row)
        groups[("__all__", "__all__", row["text_kind"])].append(row)
    for (corpus, target, text_kind), group in sorted(groups.items()):
        base = {"corpus": corpus, "target": target, "text_kind": text_kind}
        metrics: dict[str, int] = {
            "reports": len(group),
            "original_coordinate_observed_reports": sum(row["original_source_coordinate_observed"] for row in group),
            "source_parse_uncertain_reports": sum(row["source_parse_uncertain"] for row in group),
            "raw_source_coordinate_occurrences": sum(int(row["raw_source_coordinate_count"]) for row in group),
            "gp_source_coordinate_occurrences": sum(int(row["gp_source_coordinate_count"]) for row in group),
            "redacted_source_coordinate_occurrences": sum(int(row["redacted_source_coordinate_count"]) for row in group),
            "gp_coordinate_occurrences_removed": sum(int(row["gp_coordinate_occurrences_removed"]) for row in group),
            "redaction_coordinate_occurrences_removed": sum(int(row["redaction_coordinate_occurrences_removed"]) for row in group),
            "stored_primary_exact_matches": sum(row["stored_primary_exact_match"] for row in group),
            "stored_secondary_exact_matches": sum(row["stored_secondary_exact_match"] for row in group),
            "raw_runtime_path_occurrences": sum(int(row["raw_runtime_path_count"]) for row in group),
            "redacted_runtime_path_occurrences": sum(int(row["redacted_runtime_path_count"]) for row in group),
            "raw_input_filename_reports": sum(row["raw_input_filename_present"] for row in group),
            "redacted_input_placeholder_reports": sum(row["redacted_input_placeholder_present"] for row in group),
            "raw_label_marker_reports": sum(row["raw_label_marker_present"] for row in group),
            "redacted_label_placeholder_reports": sum(row["redacted_label_placeholder_present"] for row in group),
        }
        if text_kind == "asan":
            for stage in ("raw", "gp", "redacted"):
                metrics[f"{stage}_asan_initial_stack_source_coordinate_occurrences"] = sum(int(row[f"{stage}_asan_initial_stack_source_coordinate_count"] or 0) for row in group)
                metrics[f"{stage}_asan_summary_source_coordinate_occurrences"] = sum(int(row[f"{stage}_asan_summary_source_coordinate_count"] or 0) for row in group)
        for outcome in sorted({row["source_coordinate_outcome"] for row in group}):
            metrics[f"outcome:{outcome}"] = sum(row["source_coordinate_outcome"] == outcome for row in group)
        for field in ("filename_to_stored", "line_to_stored", "column_to_stored"):
            for status in sorted({row[field] for row in group}):
                metrics[f"{field}:{status}"] = sum(row[field] == status for row in group)
        for metric, value in sorted(metrics.items()):
            result.append({**base, "metric": metric, "value": int(value)})
    return result


def metric_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], int]:
    return {(row["corpus"], row["target"], row["text_kind"], row["metric"]): int(row["value"]) for row in rows}


def examples_markdown(private_rows: list[dict[str, Any]]) -> str:
    """Show category-based, de-identified examples rather than raw trace excerpts."""
    lines = [
        "# 실제 입력의 대표 사례", "",
        "원문 경로·입력 파일명·라벨은 재출력하지 않는다. source 좌표는 basename:line[:column]으로만 표기하며, `<PATH>`·`<INPUT_FILE>`·`<LABEL_PATH>`는 저장 규칙의 placeholder다.", "",
    ]
    categories = (
        ("추가 redaction으로 source 좌표 제거", lambda row: row["source_coordinate_outcome"] == "removed_at_additional_redaction"),
        ("GPTrace ASan initial stack 제거", lambda row: row["text_kind"] == "asan" and row["gp_change_kind"] == "gptrace_remove_asan_traces_initial_stack" and row["raw_source_coordinate_count"] > row["gp_source_coordinate_count"]),
        ("source 좌표가 원래 관측되지 않음", lambda row: row["source_coordinate_outcome"] == "no_original_source_coordinate_observed"),
        ("입력 파일명 hygiene", lambda row: row["raw_input_filename_present"] and row["redacted_input_placeholder_present"]),
        ("label-bearing marker hygiene", lambda row: row["raw_label_marker_present"] and not row["redacted_label_marker_present"]),
    )
    seen = set()
    for title, predicate in categories:
        candidates = [item for item in private_rows if predicate(item["row"])]
        if not candidates:
            lines += [f"## {title}", "", "실제 표본에서 해당 조합을 찾지 못했다.", ""]
            continue
        item = next((candidate for candidate in candidates if (candidate["row"]["corpus"], candidate["row"]["text_kind"]) not in seen), candidates[0])
        row = item["row"]
        seen.add((row["corpus"], row["text_kind"]))
        raw_d = safe_descriptor(item["raw"], item["report_name"], item["raw_locations"])
        gp_d = safe_descriptor(item["gp"], item["report_name"], item["gp_locations"])
        red_d = safe_descriptor(item["redacted"], item["report_name"], item["redacted_locations"])
        lines += [
            f"## {title}", "",
            f"- corpus/target/text: `{row['corpus']}` / `{row['target']}` / `{row['text_kind']}`; report ID `{row['report_id']}`",
            f"- raw → GPTrace → redaction source 좌표: `{raw_d['source_coordinate']}` → `{gp_d['source_coordinate']}` → `{red_d['source_coordinate']}`",
            f"- runtime path 수: {raw_d['runtime_path_count']} → {gp_d['runtime_path_count']} → {red_d['runtime_path_count']}; `<PATH>` 수 {red_d['path_placeholder_count']}",
            f"- raw input basename present={raw_d['input_filename_present']}; final `<INPUT_FILE>`={red_d['input_placeholder_present']}; raw label marker={raw_d['label_marker_present']}; final raw marker={red_d['label_marker_present']}",
            f"- 저장 encoder 텍스트 일치: primary={row['stored_primary_exact_match']}, joined/control={row['stored_secondary_exact_match']}",
            "",
        ]
    lines += [
        "## 유형 구분", "",
        "- source 좌표: extension을 가진 `path:line[:column]`을 구조적으로 파싱해 이전 단계의 동일 path·line·column과 대응시켰다.",
        "- runtime library path: source 좌표 토큰과 겹치지 않는 절대 POSIX path를 별도 집계했다.",
        "- crash input filename와 label marker: report basename 및 `poc_*` marker를 별도 검사했다. 이 감사 문서는 해당 실제 문자열을 다시 적지 않는다.",
    ]
    return "\n".join(lines) + "\n"


def bprime_spec() -> str:
    return """# B′ 위치 보존 수정 명세 초안

상태: **후속 검토안이며 승인·동결된 실험 규칙이 아님**. 이 문서는 B′나 B′L0를 실행하거나 기존 B를 수정하지 않는다.

## 목표와 범위

B′는 보고서에 이미 포함된 source frame의 파일 식별자와 line, 필요 시 column을 보존하면서 crash input 파일명, 정답/벤치마크 식별자와 환경 경로 접두어를 계속 제거하는 최소 변경안이다. 원본 GPTrace를 완전히 재현한다고 주장하지 않는다. 원본에 없던 소스 좌표를 source snapshot·정답·외부 매핑으로 채우지 않는다.

## 구조적 처리 규칙

1. GPTrace 전처리 뒤 text를 line/frame 단위로 스캔한다. `source-file-token:line[:column]`을 구조적으로 인식할 때만 source 좌표 처리 규칙을 적용한다. 파일 token은 C/C++ source/header 확장자와 유효한 decimal line을 만족해야 한다.
2. 식별된 source 좌표는 환경 절대 prefix를 제거한 안전한 식별자로 직렬화한다. 기본안은 `basename:line[:column]`이다. basename 충돌 위험이 있으므로 같은 평가 집합에서 서로 다른 source file이 같은 basename을 가지면 상대경로 보존안(`relative/source/path.c:line[:column]`)을 선택하거나 해당 충돌을 sidecar에 기록하고 B′ 실행을 중단한다.
3. 구조적으로 인식하지 못한 절대 path는 기존 hygiene와 같이 `<PATH>`로 치환한다. 인식 실패 source-like token은 억지로 좌표를 복원하지 않고 `<PATH>` 또는 원 규칙의 결과를 유지하며 `unparsed_source_like_token`으로 감사한다.
4. report basename은 먼저 `<INPUT_FILE>`로 치환한다. `poc_*`, `MAGMA_*`, `PDF###`, label-bearing path는 `<LABEL_PATH>` 또는 `<PATH>`로 제거한다. source 좌표 안의 filename과 crash input basename이 우연히 같을 때에는 source-coordinate parser의 구조적 판정이 우선하되, 충돌 검사 결과를 기록한다.
5. 나머지 non-source 문자와 line order는 기존 redaction 결과에서 최소 변경한다. source-coordinate token 하나를 안전한 token 하나로만 치환하며 padding, 새 frame, 외부 소스 문장, 정답 식별자를 추가하지 않는다.

## 검사

- 원 redaction과 B′를 같은 GPTrace 출력에 적용해 source coordinate 외 diff를 문자 단위로 검사한다.
- 모든 보존 좌표가 GPTrace 입력에 실제 존재했는지, 모든 raw input/label/benchmark marker가 최종 encoder text에 없는지 검사한다.
- basename 충돌·relative-path 필요 여부, parse uncertainty, source-coordinate 보존/제거 수를 report별 sidecar와 target별 집계로 남긴다.
- B′는 별도 사전 동결 protocol에서 B, BL0, B′, B′L0, LOC를 같은 report IDs·동일 선택기에서 비교해야 한다. 이번 감사의 결과만으로 위치 복원이 기존 L 이득의 원인이라고 결론내리지 않는다.
"""


def report_markdown(summary: list[dict[str, Any]], validation: dict[str, Any]) -> str:
    m = metric_lookup(summary)
    def total(kind: str, metric: str) -> int:
        return m.get(("__all__", "__all__", kind, metric), 0)
    lines = [
        "# B 입력 위치 보존 감사", "",
        "## 확정 관측", "",
        f"감사한 실제 표본은 개발 152건과 신규 평가 174건, 합계 326건이다. 각 report의 `trace`, `no_args_trace`, `asan`을 원본 저장 파일 → 현재 고정 GPTrace preprocessing replay → 추가 redaction replay → 저장 encoder text 순서로 대조했다.",
        "",
        "| text | 원본 source 좌표 발생 | GPTrace 단계 제거 발생 | 추가 redaction 제거 발생 | 저장 primary 일치 | 저장 joined/control 일치 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind in TEXT_KINDS:
        reports = total(kind, "reports")
        lines.append(
            f"| {kind} | {total(kind, 'raw_source_coordinate_occurrences')} | {total(kind, 'gp_coordinate_occurrences_removed')} | {total(kind, 'redaction_coordinate_occurrences_removed')} | {total(kind, 'stored_primary_exact_matches')}/{reports} | {total(kind, 'stored_secondary_exact_matches')}/{reports} |"
        )
    lines += [
        "",
        "`prepare_encoder_inputs.py`의 현재 POSIX absolute-path regex는 `:`를 제외 문자로 두지 않는다. 따라서 POSIX `source.c:line[:column]` 전체가 `<PATH>`로 치환될 수 있다. 이 동작은 합성 테스트와 실제 저장 입력 대조로 별도 검증했다.",
        "",
        "ASan은 GPTrace `remove_asan_traces=True`가 first ASan block의 `#` frame lines를 먼저 지운다. 그래서 다음 수치는 redaction과 구분해 기록했다: raw initial-stack source 좌표 ",
        f"{total('asan', 'raw_asan_initial_stack_source_coordinate_occurrences')} → GPTrace {total('asan', 'gp_asan_initial_stack_source_coordinate_occurrences')} → redaction {total('asan', 'redacted_asan_initial_stack_source_coordinate_occurrences')}; ",
        f"SUMMARY source 좌표는 raw {total('asan', 'raw_asan_summary_source_coordinate_occurrences')} → GPTrace {total('asan', 'gp_asan_summary_source_coordinate_occurrences')} → redaction {total('asan', 'redacted_asan_summary_source_coordinate_occurrences')}.",
        "",
        "## 문서 불일치", "",
        "`control-protocol-v1.md`는 B의 raw absolute runtime path가 `<PATH>`로 치환되더라도 line 표기는 삭제하지 않는다고 설명한다. 그러나 B를 만든 추가 redaction은 source coordinate를 구조적으로 예외 처리하지 않고 absolute POSIX path token 전체를 치환한다. 따라서 이 설명은 실제 B preprocessing 경로의 source `path:line[:column]` 보존을 뒷받침하지 못한다. C/S의 path·line 금지 규칙과 B의 실제 source-coordinate 손실은 별도 문제다.",
        "",
        "## 미확정", "",
        "개발 v4 manifest에는 GPTrace preprocessing SHA-256은 있지만 당시 `prepare_encoder_inputs.py` SHA-256이 없다. 현재 replay와 저장 v4/control-eval text의 일치는 관측됐지만, 그것만으로 당시 redaction 구현이 현재 파일과 동일했다고 확정하지 않는다. 신규 174건은 extraction manifest에 현재와 일치하는 prepare script hash가 기록되어 있어 해당 경로의 재현 근거가 더 강하다.",
        "",
        "source-coordinate parser가 인식하지 못하는 source-like token, source extension 밖의 경로, 또는 GPTrace 내부 변환으로 이미 사라진 위치는 원인과 좌표를 과도하게 추론하지 않고 CSV에서 parse-uncertain 또는 GPTrace-stage loss로 남겼다.",
        "",
        "## 가능한 결과 영향과 한계", "",
        "경로·line 정보가 B에서 사라졌다는 관측만으로 기존 L 이득이 전부 위치 복원 때문이라고 결론낼 수 없다. L에는 source code 내용, focus marker, 선택된 location 문맥 등 다른 정보가 있으며 trace 함수/주소와 source coordinates의 관계도 target마다 다르다. 인과 판단에는 별도의 사전 동결 B′ 및 B′L0 비교가 필요하다. 이번 감사는 임베딩·군집·기존 방법을 변경하거나 재실행하지 않았다.",
        "",
        "## 산출물", "",
        "report-audit.csv에는 report×text 단계별 판정, summary.csv에는 자동 집계, examples.md에는 비식별 대표 사례, Bprime_SPEC_DRAFT.md에는 후속 검토용 최소 수정 명세를 기록했다. validation.json에는 input hash 시작/종료 비교, 저장 text 재현, manifest count, 합성 regex 테스트 결과와 실행 시간이 있다.",
    ]
    return "\n".join(lines) + "\n"


def synthetic_cases(baseline: Any) -> list[dict[str, Any]]:
    cases = [
        ("absolute_posix_source", "/build/src/foo.c:12:3", "sample.bin", "<PATH>"),
        # The generic POSIX rule starts at the separator inside this relative
        # token, so it leaves the leading directory fragment but removes the
        # coordinate-bearing suffix.  This is deliberately tested, not
        # described as source-coordinate preservation.
        ("relative_source", "src/foo.c:12:3", "sample.bin", "src<PATH>"),
        ("basename_source", "foo.c:12", "sample.bin", "foo.c:12"),
        ("absolute_posix_space_adjacent", "before /build/src/foo.c:12:3 after", "sample.bin", "before <PATH> after"),
        ("path_line_column_parentheses", "call(/build/src/foo.c:12:3)", "sample.bin", "call(<PATH>)"),
        ("windows_source", r"C:\src\foo.c:12:3", "sample.bin", r"C:\src\foo.c:12:3"),
        ("input_basename", "/tmp/sample.bin", "sample.bin", None),
    ]
    rows = []
    for name, value, basename, expected in cases:
        actual = baseline.redact(value, basename)
        rows.append({
            "case": name, "input": value, "output": actual, "expected": expected,
            "expected_exact": actual == expected if expected is not None else "<INPUT_FILE>" in actual,
            "input_locations": len(scan_source_locations(value)), "output_locations": len(scan_source_locations(actual)),
        })
    return rows


def run(cli: argparse.Namespace) -> None:
    output = Path(cli.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output exists and is nonempty; use a separate run directory:{output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records, manifest_meta = input_records()
    if len(records) != 326:
        raise ValueError(f"expected observed 326 inputs, got:{len(records)}")
    before_inventory, before_digest = source_input_inventory(records)
    baseline = import_file_module("audit_baseline_redact", WORK / "prepare_encoder_inputs.py")
    pre = import_file_module("audit_gptrace_preprocessing", GP_PREPROCESSING)
    current_prepare_hash, current_pre_hash = sha256_file(WORK / "prepare_encoder_inputs.py"), sha256_file(GP_PREPROCESSING)
    v4_manifest = manifest_meta["development_manifest"]
    new_extraction = load_json(NEW / "extraction-manifest.json")
    audit_rows, private = [], []
    for index, record in enumerate(records, 1):
        rows, confidential = audit_record(record, pre, baseline, argparse)
        audit_rows.extend(rows); private.extend(confidential)
        if index % 50 == 0:
            print(f"audited {index}/{len(records)}", flush=True)
    summaries = summary_rows(audit_rows)
    synthetic = synthetic_cases(baseline)
    after_inventory, after_digest = source_input_inventory(records)
    input_files = {
        "독립검토_후속작업_순서.md": ROOT / "독립검토_후속작업_순서.md",
        "work/prepare_encoder_inputs.py": WORK / "prepare_encoder_inputs.py",
        "work/new_project_evaluation_v2.py": WORK / "new_project_evaluation_v2.py",
        "gptrace/preprocessing.py": GP_PREPROCESSING,
        "outputs/revised/encoder-inputs-v4.jsonl": REVISED / "encoder-inputs-v4.jsonl",
        "outputs/revised/control-eval-encoder-inputs-v1.jsonl": REVISED / "control-eval-encoder-inputs-v1.jsonl",
        "outputs/revised/contexts-v4.jsonl": REVISED / "contexts-v4.jsonl",
        "outputs/revised/encoder-input-v4-manifest.json": REVISED / "encoder-input-v4-manifest.json",
        "outputs/revised/control-eval-input-manifest-v1.json": REVISED / "control-eval-input-manifest-v1.json",
        "outputs/revised/control-protocol-v1.md": REVISED / "control-protocol-v1.md",
        "outputs/revised/control-protocol-v1.json": REVISED / "control-protocol-v1.json",
        "outputs/new-project-evaluation-v2/protocol.json": NEW / "protocol.json",
        "outputs/new-project-evaluation-v2/control-protocol-v1.md": NEW / "control-protocol-v1.md",
        "outputs/new-project-evaluation-v2/control-protocol-v1.json": NEW / "control-protocol-v1.json",
        "outputs/new-project-evaluation-v2/input-manifest.json": NEW / "input-manifest.json",
        "outputs/new-project-evaluation-v2/extraction-manifest.json": NEW / "extraction-manifest.json",
        "outputs/new-project-evaluation-v2/baseline-inputs.jsonl": NEW / "baseline-inputs.jsonl",
        "outputs/new-project-evaluation-v2/encoder-inputs.jsonl": NEW / "encoder-inputs.jsonl",
    }
    input_hashes_before = {name: sha256_file(path) for name, path in input_files.items()}
    validation = {
        "schema_version": "b-location-audit-validation-v1",
        "checks": {
            "manifest_counts_152_plus_174_equals_326": len(records) == 326,
            "development_target_counts": dict(collections.Counter(row["target"] for row in records if row["corpus"] == "development_v4")) == {"freetype__char2svg": 51, "poppler__pdfimages": 40, "soxmp3__sox": 61},
            "new_target_counts": dict(collections.Counter(row["target"] for row in records if row["corpus"] == "new_project_v2")) == {"libtiff__tiff2pdf": 30, "libtiff__tiffcp": 60, "libxml2__libxml2_xml_read_memory_fuzzer": 10, "libxml2__xmllint": 74},
            "all_saved_primary_texts_compared": len(audit_rows) == len(records) * len(TEXT_KINDS),
            "all_saved_secondary_texts_compared": len(audit_rows) == len(records) * len(TEXT_KINDS),
            "all_synthetic_cases_pass": all(row["expected_exact"] is True for row in synthetic),
            "current_gptrace_hash_matches_v4_manifest": current_pre_hash == v4_manifest["preprocessing_sha256"],
            "current_prepare_hash_matches_new_extraction_manifest": current_prepare_hash == new_extraction["implementation_files"]["work/prepare_encoder_inputs.py"],
            "selected_raw_report_trace_inputs_unchanged": before_digest == after_digest and before_inventory == after_inventory,
            "no_embedding_or_clustering_executed": True,
        },
        "synthetic_cases": synthetic,
        "input_inventory_before_sha256": before_digest, "input_inventory_after_sha256": after_digest,
        "input_file_hashes_before": input_hashes_before,
        "current_prepare_sha256": current_prepare_hash, "current_gptrace_preprocessing_sha256": current_pre_hash,
        "development_historical_redaction_version": "not recorded in v4 manifest",
        "new_historical_redaction_version": new_extraction["implementation_files"]["work/prepare_encoder_inputs.py"],
        "saved_primary_replay_mismatch_count": sum(not row["stored_primary_exact_match"] for row in audit_rows),
        "saved_secondary_replay_mismatch_count": sum(not row["stored_secondary_exact_match"] for row in audit_rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    validation["status"] = "passed" if all(validation["checks"].values()) else "failed"
    input_manifest = {
        "schema_version": "b-location-audit-input-manifest-v1",
        "scope": "read-only preprocessing audit; no embedding, clustering, or method change",
        "counts": {
            "development_v4": dict(sorted(collections.Counter(row["target"] for row in records if row["corpus"] == "development_v4").items())),
            "new_project_v2": dict(sorted(collections.Counter(row["target"] for row in records if row["corpus"] == "new_project_v2").items())),
            "total": len(records),
        },
        "input_files_sha256": input_hashes_before,
        "selected_raw_report_trace_inventory_sha256_before": before_digest,
        "selected_raw_report_trace_inventory": before_inventory,
        "code_versions": {
            "current_prepare_encoder_inputs_sha256": current_prepare_hash,
            "current_gptrace_preprocessing_sha256": current_pre_hash,
            "v4_manifest_gptrace_preprocessing_sha256": v4_manifest["preprocessing_sha256"],
            "v4_prepare_encoder_inputs_sha256": None,
            "new_extraction_prepare_encoder_inputs_sha256": new_extraction["implementation_files"]["work/prepare_encoder_inputs.py"],
        },
        "reconstruction_boundary": {
            "development_v4": "current replay is compared to saved text, but v4 manifest lacks prepare_encoder_inputs.py hash; historical redaction implementation remains unconfirmed",
            "new_project_v2": "new extraction manifest records current prepare_encoder_inputs.py hash; replay is code-version-supported as well as text-matched",
        },
    }
    write_json_new(output / "input-manifest.json", input_manifest)
    write_csv_new(output / "report-audit.csv", audit_rows)
    write_csv_new(output / "summary.csv", summaries)
    write_new(output / "examples.md", examples_markdown(private))
    write_new(output / "Bprime_SPEC_DRAFT.md", bprime_spec())
    write_new(output / "REPORT.md", report_markdown(summaries, validation))
    write_new(output / "HANDOFF.md", "# Handoff\n\nCompleted the read-only B location-preprocessing audit across 152 development and 174 new-project reports. Existing inputs, code, caches, progress files, and experiment outputs were not modified. See REPORT.md for the bounded conclusion, validation.json for reproduction evidence, and Bprime_SPEC_DRAFT.md for the non-approved follow-up proposal.\n")
    write_new(output / "FAILURE_LOG.md", "# Failure log\n\nNo unresolved audit execution failure. The development v4 manifest does not record a `prepare_encoder_inputs.py` hash; this is a provenance limitation, not silently treated as historical-code evidence. It is recorded as unconfirmed in input-manifest.json, validation.json, and REPORT.md.\n")
    input_hashes_after = {name: sha256_file(path) for name, path in input_files.items()}
    validation["input_file_hashes_after"] = input_hashes_after
    validation["checks"]["read_input_files_unchanged"] = input_hashes_before == input_hashes_after
    validation["status"] = "passed" if all(validation["checks"].values()) else "failed"
    write_json_new(output / "validation.json", validation)
    if validation["status"] != "passed":
        raise ValueError(json.dumps(validation, indent=2))
    print(json.dumps({"status": validation["status"], "reports": len(records), "rows": len(audit_rows), "elapsed_seconds": validation["elapsed_seconds"]}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("command", choices=("run",))
    result.add_argument("--output", default=str(OUTPUT_DEFAULT))
    return result


def main() -> None:
    cli = parser().parse_args()
    run(cli)


if __name__ == "__main__":
    main()

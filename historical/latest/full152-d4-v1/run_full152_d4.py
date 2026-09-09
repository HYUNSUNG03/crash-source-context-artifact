"""Materialize and extract the fixed-D=4, label-free panel for the frozen 152.

This runner deliberately reuses the already-frozen, line-preserving Stage 2
source directory and its graph.  It never rebuilds the CPG and it never joins
labels to the output.  A UAF row without report-specific dynamic lifetime
evidence is retained only as an explicit static-only/abstaining result.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(r"/RESEARCH_WORKSPACES\2026-09-08\new-chat-2")
OLD = Path(r"/RESEARCH_WORKSPACES\2026-09-05\new-chat-2")
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/full152-d4-readiness-v1"
STAGE2 = ROOT / "work/stage2-d4-v1"
UAF_FTSTREAM = ROOT / "work/d1/uaf/source/src/base/ftstream.c"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def asan_type(text: str) -> str | None:
    match = re.search(r"AddressSanitizer:\s+([^\s]+)", text or "", flags=re.I)
    return match.group(1) if match else None


def prefix(source_id: str) -> str:
    if source_id.startswith("freetype-"):
        return "freetype"
    if source_id.startswith("poppler-"):
        return "poppler"
    if source_id.startswith("sox-"):
        return "sox"
    return source_id


def same_location(frame: dict, location: dict) -> bool:
    return (
        frame.get("source_id") == location.get("source_id")
        and frame.get("source_relative_path") == location.get("source_relative_path")
        and frame.get("line") == location.get("line")
        and frame.get("function_hint") == location.get("function_hint")
    )


def parse_uaf_facts(text: str) -> dict:
    """Keep report-specific ASan facts; do not infer object identity from them."""
    access = re.search(r"\b(READ|WRITE) of size (\d+) at (0x[0-9a-fA-F]+)", text)
    region = re.search(r"located (\d+) bytes inside of (\d+)-byte region \[(0x[0-9a-fA-F]+),(0x[0-9a-fA-F]+)\)", text)
    freed = re.search(r"freed by thread .*? here:\n(?P<frames>(?:\s*#\d+[^\n]*\n?)+)", text, flags=re.S)
    allocated = re.search(r"previously allocated by thread .*? here:\n(?P<frames>(?:\s*#\d+[^\n]*\n?)+)", text, flags=re.S)
    if not all((access, region, freed, allocated)):
        raise ValueError("incomplete frozen ASan UAF evidence")
    return {
        "access_kind": access.group(1), "access_size": int(access.group(2)),
        "access_address": access.group(3), "fault_offset": int(region.group(1)),
        "freed_region_size": int(region.group(2)), "freed_region_start": region.group(3),
        "freed_region_end": region.group(4),
        "freed_frames": [line.strip() for line in freed.group("frames").splitlines() if line.strip()],
        "allocated_frames": [line.strip() for line in allocated.group("frames").splitlines() if line.strip()],
        "asan_proves_accessed_allocation_was_freed": True,
    }


def render_d4(row: dict) -> str:
    lines = [f"[D4_ENDPOINT_STATUS] {row.get('endpoint_status')}", "[SELECTED_STATIC_DEFINITIONS]"]
    endpoints = row.get("selected_definition_endpoints") or []
    lines.extend(
        f"{item.get('origin_role')}:{item.get('origin_name')} -> {item.get('code')} @ "
        f"{item.get('source_file')}:{item.get('line')} hops={item.get('total_hops')} "
        f"confidence={item.get('confidence')}" for item in endpoints
    ) if endpoints else lines.append("<none>")
    lines.append("[ARGUMENT_BOUNDARIES]")
    boundaries = row.get("argument_boundary_candidates") or []
    lines.extend(json.dumps(item, sort_keys=True, separators=(",", ":")) for item in boundaries) if boundaries else lines.append("<none>")
    lines.append("[CONSTRAINT_CONTEXT]")
    constraints = row.get("constraint_context") or []
    lines.extend(f"{item.get('code')} @ line {item.get('line')}" for item in constraints) if constraints else lines.append("<none>")
    lines.append("[UAF_STATIC_SIDE_EFFECT_CANDIDATES]")
    effects = row.get("uaf_side_effect_candidates") or []
    lines.extend(json.dumps(item, sort_keys=True, separators=(",", ":")) for item in effects) if effects else lines.append("<none>")
    if row.get("uaf_dynamic_evidence_status") == "unavailable":
        facts = row["uaf_static_lifetime_evidence"]["asan_facts"]
        lines.extend([
            "[UAF_DYNAMIC_FACTS]",
            f"{facts['access_kind']} size={facts['access_size']} offset={facts['fault_offset']} freed_region_size={facts['freed_region_size']}",
            "accessed_allocation_was_freed=true",
            "[UAF_FAULT_OPERAND]",
            "FT_NEXT_USHORT( p ) -> p",
            "[UAF_OBJECT_PROVENANCE_D4]",
        ])
        lines.extend(
            f"hop={edge['hop']} {edge['kind']}: {edge['from']} -> {edge['to']} @ {edge['file']}:{edge['line']}"
            for edge in row["uaf_static_lifetime_evidence"]["static_path"]
        )
        lines.extend([
            "[UAF_FREE_EVENT_STATUS] static_same_field_candidate; exact_dynamic_callsite=false; root_cause=false; alias_identity=false",
            "[UAF_INPUT_OBJECT_IDENTITY] omitted_for_panel_fairness; confirmed=false",
            "[UAF_DYNAMIC_EVIDENCE] report_specific_ASan_facts=true; free_stack_ends_at_wrapper=true",
        ])
    lines.append("[CLAIM_CEILING] static_candidate_only; runtime_path=false; root_cause=false; alias_identity=false")
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / "work/l-grid-v1/all152/protocol.json"
    contexts_path = OLD / "outputs/revised/contexts-v4.jsonl"
    frames_path = OLD / "outputs/l3-development-v1/contexts.jsonl"
    bases_path = OLD / "outputs/frame-function-v1/numeric-base-inputs.jsonl"
    graph_path = STAGE2 / "graph.json"
    source_root = STAGE2 / "source"
    uaf_evidence_path = ROOT / "outputs/full152-uaf-v1/lifetime-evidence-label-free.jsonl"
    uaf_texts_path = ROOT / "outputs/full152-uaf-v1/uaf-d4-texts-label-free.jsonl"
    ids = list(json.loads(protocol_path.read_text(encoding="utf-8"))["sample_ids"])
    contexts = {row["report_id"]: row for row in read_jsonl(contexts_path)}
    frames = {row["report_id"]: row for row in read_jsonl(frames_path)}
    bases = {row["report_id"]: row for row in read_jsonl(bases_path)}

    requests, audit = [], []
    for report_id in ids:
        context, frame_row, base = contexts.get(report_id), frames.get(report_id), bases.get(report_id)
        if not (context and frame_row and base):
            audit.append({"report_id": report_id, "eligible": False, "reason": "frozen_input_missing"})
            continue
        location = context.get("selected_location") or {}
        matched = [frame for frame in frame_row.get("frames", []) if same_location(frame, location)]
        relative = f"{prefix(str(location.get('source_id', '')))}/{location.get('source_relative_path', '')}".replace("\\", "/")
        source = source_root / relative
        diagnostic = (base.get("texts") or {}).get("asan", "")
        error = asan_type(diagnostic)
        source_line = None
        if source.is_file() and isinstance(location.get("line"), int):
            lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
            if 0 < location["line"] <= len(lines):
                source_line = lines[location["line"] - 1].strip()
        focused = [line.removeprefix("FOCUS: ").strip() for line in (matched[0].get("text", "").splitlines() if len(matched) == 1 else []) if line.startswith("FOCUS:")]
        reason = None
        if not source.is_file():
            reason = "existing_graph_source_missing"
        elif len(matched) != 1:
            reason = "focus_frame_not_unique"
        elif len(focused) != 1 or focused[0] != source_line:
            reason = "focus_line_mismatch"
        elif not error:
            reason = "asan_type_missing"
        if reason:
            audit.append({"report_id": report_id, "eligible": False, "reason": reason, "file": relative})
            continue
        focus_index = next(index for index, frame in enumerate(frame_row["frames"]) if same_location(frame, location))
        callers = [{"function": frame.get("function_hint"), "line": frame.get("line")} for frame in frame_row["frames"][focus_index + 1:focus_index + 4]]
        requests.append({
            "id": report_id, "file": relative, "function": location.get("function_hint"),
            "line": location.get("line"), "column": location.get("column"), "asan_type": error,
            "call_depth": 3, "stack_caller_frames": callers,
        })
        audit.append({"report_id": report_id, "eligible": True, "reason": "frozen_asan_exact_stack_line_preserving_graph_source", "file": relative, "asan_type": error, "saved_caller_frames": len(callers)})

    requests_path = OUT / "requests.json"
    audit_path = OUT / "request-audit.jsonl"
    requests_path.write_text(json.dumps(requests, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    audit_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in audit), encoding="utf-8", newline="\n")
    if len(requests) != len(ids):
        raise RuntimeError(f"request gate failed: {len(requests)}/{len(ids)}")

    candidates_path = OUT / "candidate-contexts.jsonl"
    endpoints_path = OUT / "selected-endpoints.jsonl"
    subprocess.run([sys.executable, str(ROOT / "work/d1/extract_d1.py"), "--graph", str(graph_path), "--source", str(source_root), "--requests", str(requests_path), "--output", str(candidates_path), "--max-rd-depth", "4"], check=True)
    subprocess.run([sys.executable, str(ROOT / "work/d1/select_d_endpoints.py"), "--input", str(candidates_path), "--source", str(source_root), "--output", str(endpoints_path)], check=True)

    endpoint_rows = {row["id"]: row for row in read_jsonl(endpoints_path)}
    candidate_rows = {row["id"]: row for row in read_jsonl(candidates_path)}
    uaf_evidence = {row["report_id"]: row for row in read_jsonl(uaf_evidence_path)}
    uaf_texts = {row["report_id"]: row for row in read_jsonl(uaf_texts_path)}
    label_free, rejected = [], []
    for request in requests:
        row = endpoint_rows[request["id"]]
        candidate = candidate_rows[request["id"]]
        is_uaf = "use-after-free" in request["asan_type"].lower()
        if not row.get("endpoint_status"):
            row = dict(row)
            row["endpoint_status"] = f"extractor_abstain_{candidate.get('status', 'unknown')}"
        if is_uaf:
            row = dict(row)
            if request["id"] not in uaf_evidence or request["id"] not in uaf_texts:
                raise RuntimeError(f"missing validated UAF evidence for {request['id']}")
            # Consume the separately validated, report-specific ASan evidence.
            # It intentionally marks every dynamic free/root-cause/alias/identity
            # claim false, even where a single legacy report had extra evidence.
            row["uaf_dynamic_evidence_status"] = "integrated_validated_static_only"
            row["endpoint_status"] = "uaf_static_only_dynamic_evidence_unavailable"
        text = render_d4(row)
        if is_uaf:
            text += "\n" + uaf_texts[request["id"]]["d4_uaf_text"]
        label_free.append({
            "report_id": request["id"], "d4_text": text, "d4_text_sha256": text_sha(text),
            "endpoint_status": row.get("endpoint_status"), "request_sha256": text_sha(json.dumps(request, sort_keys=True, separators=(",", ":"))),
            "extraction_mode": "report_specific_frozen_asan_stack_line_preserving_source_graph",
            "max_causal_depth": 4, "uaf_dynamic_evidence_status": row.get("uaf_dynamic_evidence_status", "not_applicable"),
            "uaf_evidence_sha256": text_sha(json.dumps(uaf_evidence[request["id"]], sort_keys=True, separators=(",", ":"))) if is_uaf else None,
        })
        if candidate.get("status") != "candidate_context":
            rejected.append({"report_id": request["id"], "reason": candidate.get("status")})
    panel_path = OUT / "d4-texts-label-free.jsonl"
    rejected_path = OUT / "extraction-abstentions.jsonl"
    panel_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in label_free), encoding="utf-8", newline="\n")
    rejected_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rejected), encoding="utf-8", newline="\n")
    graph_bytes = graph_path.stat().st_size
    validation = {
        "status": "pass" if len(label_free) == len(ids) else "failed",
        "frozen_reports": len(ids), "requests_available": len(requests), "d4_rows": len(label_free),
        "candidate_contexts": sum(row.get("status") == "candidate_context" for row in candidate_rows.values()),
        "extractor_abstentions": len(rejected), "endpoint_status_counts": dict(sorted(Counter(row["endpoint_status"] for row in label_free).items())),
        "uaf_dynamic_evidence_integrated_static_only": sum(row["uaf_dynamic_evidence_status"] == "integrated_validated_static_only" for row in label_free),
        "max_causal_depth": 4, "D_depth_reoptimized": False, "labels_loaded": False, "labels_in_d4_text": False,
        "graph_reused": {"path": str(graph_path), "sha256": sha(graph_path), "bytes": graph_bytes, "no_rebuild": True},
        "estimated_scope": {"graph_builds_launched": 0, "report_extractions": len(requests), "source_documents_in_existing_graph": 18},
        "inputs": {name: {"path": str(path), "sha256": sha(path)} for name, path in {"protocol": protocol_path, "contexts": contexts_path, "frames": frames_path, "base_inputs": bases_path, "validated_uaf_evidence": uaf_evidence_path, "validated_uaf_texts": uaf_texts_path}.items()},
        "outputs": {name: {"path": str(path), "sha256": sha(path)} for name, path in {"requests": requests_path, "audit": audit_path, "candidates": candidates_path, "endpoints": endpoints_path, "label_free_d4": panel_path, "abstentions": rejected_path}.items()},
        "claim_limit": "All rows are static candidates from frozen ASan/stack and a line-preserving source graph. No row proves runtime path, root cause, or alias identity. UAF rows without report-specific dynamic lifetime evidence are explicitly static-only or abstaining.",
    }
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: validation[key] for key in ("status", "frozen_reports", "candidate_contexts", "extractor_abstentions", "endpoint_status_counts", "uaf_dynamic_evidence_integrated_static_only")}, indent=2))
    return 0 if validation["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

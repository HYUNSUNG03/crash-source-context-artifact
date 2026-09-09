"""Build the corrected fixed-D=4, label-free v2 panel from frozen v1 extraction.

v2 changes only rendering/selection policy: endpoint candidates must originate
from a memory-semantic causal operand, numeric literals are never endpoints,
and each text contains the retained fault expression and operands.  No graph is
rebuilt and labels are never read.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(r"/RESEARCH_WORKSPACES\2026-09-08\new-chat-2")
V1 = ROOT / "outputs/full152-d4-readiness-v1"
OUT = ROOT / "outputs/full152-d4-readiness-v2"
UAF = ROOT / "outputs/full152-uaf-v1"
FIXED32 = ROOT / "outputs/ld-microaudit-v4/protocol.json"

NUMERIC_ASSIGNMENT = re.compile(
    r"=\s*(?:[-+]?\d+[uUlL]*|0[xX][0-9a-fA-F]+[uUlL]*)\s*;?\s*$"
)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fault_sections(candidate: dict) -> list[str]:
    if candidate.get("status") != "candidate_context":
        return ["[FAULTING_EXPRESSION] <abstain:" + str(candidate.get("status")) + ">", "[CAUSAL_OPERANDS]", "<none>"]
    expression = candidate.get("faulting_expression_top1") or {}
    expression_text = (
        f"{expression.get('kind')}:{expression.get('code')} @ "
        f"line {expression.get('line')} column {expression.get('column')}"
    )
    operands = candidate.get("causal_operand_candidates") or []
    lines = ["[FAULTING_EXPRESSION] " + expression_text, "[CAUSAL_OPERANDS]"]
    lines.extend(
        f"{item.get('role')}:{item.get('code')} @ line {item.get('line')} "
        f"reason={item.get('selection_reason')}" for item in operands
    )
    return lines if operands else [*lines, "<none>"]


def render(row: dict, candidate: dict, uaf_text: str | None) -> str:
    lines = [f"[D4_ENDPOINT_STATUS] {row['endpoint_status']}", *fault_sections(candidate), "[SELECTED_STATIC_DEFINITIONS]"]
    endpoints = row.get("selected_definition_endpoints") or []
    lines.extend(
        f"{item.get('origin_role')}:{item.get('origin_name')} -> {item.get('code')} @ "
        f"{item.get('source_file')}:{item.get('line')} hops={item.get('total_hops')} confidence={item.get('confidence')}"
        for item in endpoints
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
    lines.append("[CLAIM_CEILING] static_candidate_only; runtime_path=false; root_cause=false; alias_identity=false")
    if uaf_text:
        lines.append(uaf_text)
    return "\n".join(lines)


def cohort_summary(values: list[dict], fixed_ids: set[str]) -> dict:
    return {
        "rows": len(values),
        "fixed32_rows": sum(row["report_id"] in fixed_ids for row in values),
        "candidate_contexts": sum(row["candidate_status"] == "candidate_context" for row in values),
        "fault_expression_rendered": sum("[FAULTING_EXPRESSION]" in row["d4_text"] for row in values),
        "causal_operands_rendered": sum("[CAUSAL_OPERANDS]" in row["d4_text"] for row in values),
        "endpoint_status_counts": dict(sorted(Counter(row["endpoint_status"] for row in values).items())),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates_path = V1 / "candidate-contexts.jsonl"
    endpoints_path = V1 / "selected-endpoints.jsonl"
    requests_path = V1 / "requests.json"
    candidates = {row["id"]: row for row in rows(candidates_path)}
    endpoints = {row["id"]: row for row in rows(endpoints_path)}
    requests = json.loads(requests_path.read_text(encoding="utf-8"))
    uaf_texts = {row["report_id"]: row for row in rows(UAF / "uaf-d4-texts-label-free.jsonl")}
    uaf_evidence = {row["report_id"]: row for row in rows(UAF / "lifetime-evidence-label-free.jsonl")}
    fixed_ids = set(json.loads(FIXED32.read_text(encoding="utf-8"))["sample_ids"])

    endpoint_filter_counts = Counter()
    selected_v2, panel = [], []
    for request in requests:
        report_id = request["id"]
        candidate = candidates[report_id]
        original = endpoints[report_id]
        is_uaf = "use-after-free" in request["asan_type"].lower()
        operand_roles = {item.get("role") for item in candidate.get("causal_operand_candidates", [])}
        kept = []
        for endpoint in original.get("selected_definition_endpoints") or []:
            if NUMERIC_ASSIGNMENT.search(endpoint.get("code", "")):
                endpoint_filter_counts["numeric_literal_assignment_excluded"] += 1
                continue
            if endpoint.get("origin_role") not in operand_roles:
                endpoint_filter_counts["non_causal_operand_role_excluded"] += 1
                continue
            kept.append(endpoint)
        row = dict(original)
        row["id"] = report_id
        row["selected_definition_endpoints"] = kept
        row["endpoint_filter_policy"] = "v2_actual_causal_operand_roles_only_and_numeric_literal_rejection"
        if is_uaf:
            if report_id not in uaf_texts or report_id not in uaf_evidence:
                raise RuntimeError(f"validated UAF evidence missing: {report_id}")
            row["endpoint_status"] = "uaf_static_only_dynamic_evidence_unavailable"
            row["uaf_evidence_sha256"] = text_sha(json.dumps(uaf_evidence[report_id], sort_keys=True, separators=(",", ":")))
            uaf_text = uaf_texts[report_id]["d4_uaf_text"]
        elif candidate.get("status") != "candidate_context":
            row["endpoint_status"] = "extractor_abstain_" + str(candidate.get("status"))
            uaf_text = None
        elif kept:
            row["endpoint_status"] = "selected_static_definition"
            uaf_text = None
        elif row.get("argument_boundary_candidates"):
            row["endpoint_status"] = "argument_boundary_only"
            uaf_text = None
        else:
            row["endpoint_status"] = "no_static_endpoint"
            uaf_text = None
        text = render(row, candidate, uaf_text)
        selected_v2.append(row)
        panel.append({
            "report_id": report_id, "candidate_status": candidate.get("status"), "endpoint_status": row["endpoint_status"],
            "d4_text": text, "d4_text_sha256": text_sha(text),
            "request_sha256": text_sha(json.dumps(request, sort_keys=True, separators=(",", ":"))),
            "extraction_mode": "report_specific_frozen_asan_stack_line_preserving_source_graph",
            "max_causal_depth": 4, "uaf_evidence_integrated": is_uaf,
        })

    endpoints_out = OUT / "selected-endpoints-v2.jsonl"
    panel_out = OUT / "d4-texts-label-free-v2.jsonl"
    endpoints_out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected_v2), encoding="utf-8", newline="\n")
    panel_out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in panel), encoding="utf-8", newline="\n")
    before = sum(len(row.get("selected_definition_endpoints") or []) for row in endpoints.values())
    after = sum(len(row.get("selected_definition_endpoints") or []) for row in selected_v2)
    full = cohort_summary(panel, fixed_ids)
    fixed = cohort_summary([row for row in panel if row["report_id"] in fixed_ids], fixed_ids)
    retained = [item for row in selected_v2 for item in row.get("selected_definition_endpoints") or []]
    retained_role_gate = all(
        item.get("origin_role") in {operand.get("role") for operand in candidates[row["id"]].get("causal_operand_candidates", [])}
        for row in selected_v2 for item in row.get("selected_definition_endpoints") or []
    )
    retained_numeric_gate = not any(NUMERIC_ASSIGNMENT.search(item.get("code", "")) for item in retained)
    checks = {
        "all_152_rows": len(panel) == 152 and len({row["report_id"] for row in panel}) == 152,
        "fixed32_complete": fixed["rows"] == 32,
        "fault_and_operand_sections_all_rows": full["fault_expression_rendered"] == 152 and full["causal_operands_rendered"] == 152,
        "selected_endpoints_are_causal_operands": retained_role_gate,
        "selected_endpoints_exclude_numeric_literals": retained_numeric_gate,
        "uaf_evidence_integrated": sum(row["uaf_evidence_integrated"] for row in panel) == 10,
        "uaf_claims_static_only": all(not evidence["claims"][claim] for evidence in uaf_evidence.values() for claim in ("exact_dynamic_free_callsite_proven", "root_cause_proven", "alias_identity_proven", "input_object_identity_confirmed")),
        "label_free": not any("poc_" in row["d4_text"] for row in panel),
    }
    validation = {
        "schema_version": "full152-fixed-d4-v2", "status": "pass" if all(checks.values()) else "failed", "max_causal_depth": 4,
        "D_depth_reoptimized": False, "labels_loaded": False, "labels_in_d4_text": False,
        "full152": full, "fixed32": fixed,
        "endpoint_count_before_v2_filter": before, "endpoint_count_after_v2_filter": after,
        "endpoint_filter_counts": dict(sorted(endpoint_filter_counts.items())),
        "uaf_rows": sum(row["uaf_evidence_integrated"] for row in panel),
        "checks": checks,
        "inputs": {name: {"path": str(path), "sha256": sha(path)} for name, path in {"v1_candidates": candidates_path, "v1_endpoints": endpoints_path, "requests": requests_path, "validated_uaf_evidence": UAF / "lifetime-evidence-label-free.jsonl", "validated_uaf_texts": UAF / "uaf-d4-texts-label-free.jsonl", "fixed32_protocol": FIXED32}.items()},
        "outputs": {name: {"path": str(path), "sha256": sha(path)} for name, path in {"selected_endpoints_v2": endpoints_out, "label_free_d4_v2": panel_out}.items()},
        "claim_limit": "Static report-specific candidates only. UAF facts are report-specific but the recovered four-hop free path is static; no exact dynamic free callsite, root cause, alias identity, or input-object identity is claimed.",
    }
    (OUT / "validation-v2.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": validation["status"], "full152": full, "fixed32": fixed, "endpoint_filter_counts": validation["endpoint_filter_counts"], "uaf_rows": validation["uaf_rows"]}, indent=2))


if __name__ == "__main__":
    main()

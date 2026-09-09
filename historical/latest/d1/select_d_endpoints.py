#!/usr/bin/env python3
"""Select conservative D=4 causal endpoints from extractor candidate context."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


IDENT = re.compile(r"[A-Za-z_]\w*")
ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*(?:\s*(?:->|\.)\s*[A-Za-z_]\w*)*)\s*(=|\+=|-=)\s*(.+?)\s*;?\s*$", re.S)
SIMPLE_SOURCE = re.compile(r"^\s*(?:\([^()]+\)\s*)*[A-Za-z_]\w*(?:\s*(?:->|\.)\s*[A-Za-z_]\w*)*\s*$")
ALLOWED_TRANSFORM_CALLS = {"strlen", "sizeof"}
IGNORED_IDENTIFIERS = {
    "const", "char", "short", "int", "long", "signed", "unsigned", "size_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "void", "struct", "sizeof",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "").rstrip(";")


def identifiers(text: str) -> set[str]:
    return {name for name in IDENT.findall(text or "") if name not in IGNORED_IDENTIFIERS}


def assignment_parts(code: str):
    match = ASSIGN.match(code or "")
    if not match:
        return None
    lhs, operator, rhs = match.groups()
    lhs_root = IDENT.match(lhs.strip()).group(0)
    return lhs.strip(), lhs_root, operator, rhs.strip().rstrip(";")


def source_match(source_root: Path, line: int, code: str, cache: dict[Path, list[str]]):
    wanted = compact(code)
    if not wanted or not line:
        return None
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".c", ".h", ".cc", ".cpp"}:
            continue
        if path not in cache:
            try:
                cache[path] = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                cache[path] = []
        lines = cache[path]
        if 0 < line <= len(lines) and wanted in compact(lines[line - 1]):
            return str(path.relative_to(source_root)).replace("\\", "/")
    return None


def is_self_update(lhs_root: str, operator: str, rhs: str) -> bool:
    rhs_ids = identifiers(rhs)
    return operator != "=" or (rhs_ids == {lhs_root})


def is_loop_state(role: str, lhs_root: str, rhs: str, code: str, constraints: list[dict]) -> bool:
    if role not in {"index", "control"}:
        return False
    in_guard = any(lhs_root in identifiers(item.get("code", "")) for item in constraints)
    literal_init = bool(re.fullmatch(r"[-+]?\d+[uUlL]*", rhs.strip()))
    update = "++" in code or "--" in code
    return in_guard and (literal_init or update)


def endpoint_class(rhs: str):
    if SIMPLE_SOURCE.fullmatch(rhs):
        return "direct_transfer", 3
    calls = {
        name for name in IDENT.findall(rhs)
        if re.search(rf"\b{re.escape(name)}\s*\(", rhs)
    }
    if calls.issubset(ALLOWED_TRANSFORM_CALLS) and identifiers(rhs):
        return "bounded_transform", 3
    return None, 0


def reject(reason_counts, reason):
    reason_counts[reason] += 1


def definition_candidates(row: dict, source_root: Path, constraints: list[dict]):
    cache: dict[Path, list[str]] = {}
    rejected = collections.Counter()
    candidates = []
    uaf = "use-after-free" in row.get("asan_type", "").lower()
    if uaf:
        return candidates, rejected

    # Interprocedural definitions: only exact saved stack paths are eligible.
    exact_scope = row.get("argument_path_scope") == "saved_stack_only" and row.get("stack_caller_frames") is not None
    for path in row.get("argument_paths", []):
        if not exact_scope:
            reject(rejected, "non_exact_stack_scope")
            continue
        arg_ids = identifiers(path.get("argument_code", ""))
        for item in path.get("caller_reaching_definitions", []):
            site = item.get("definition_site")
            if not site:
                continue
            parts = assignment_parts(site.get("code", ""))
            if not parts:
                reject(rejected, "not_assignment")
                continue
            lhs, lhs_root, operator, rhs = parts
            if item.get("name") != lhs_root or lhs_root not in arg_ids:
                reject(rejected, "lhs_target_mismatch")
                continue
            total_hops = path.get("hop", 0) + item.get("depth", 0)
            if total_hops > row.get("max_causal_depth", 4):
                reject(rejected, "over_depth_cap")
                continue
            if is_self_update(lhs_root, operator, rhs):
                reject(rejected, "self_update")
                continue
            if is_loop_state(path.get("origin_role", ""), lhs_root, rhs, site.get("code", ""), constraints):
                reject(rejected, "loop_state")
                continue
            kind, class_score = endpoint_class(rhs)
            if not kind:
                reject(rejected, "unsupported_transform")
                continue
            matched_file = source_match(source_root, site.get("line"), site.get("code", ""), cache)
            if not matched_file:
                reject(rejected, "definition_not_source_visible")
                continue
            candidates.append({
                "origin_role": path.get("origin_role"),
                "origin_name": path.get("origin_name"),
                "target_identifier": lhs_root,
                "endpoint_class": kind,
                "confidence": "static_candidate",
                "total_hops": total_hops,
                "path_hop": path.get("hop"),
                "definition_hop": item.get("depth"),
                "caller": path.get("caller"),
                "call_line": path.get("call_line"),
                "argument_code": path.get("argument_code"),
                "line": site.get("line"),
                "code": site.get("code"),
                "source_file": matched_file,
                "score": class_score + 2,
            })

    # Intraprocedural definitions use the same structural LHS gate.
    seed_roles = collections.defaultdict(set)
    for seed in [*row.get("seed_identifiers", []), *row.get("control_seed_identifiers", [])]:
        seed_roles[seed.get("name")].add(seed.get("origin_role"))
    for item in row.get("reaching_definitions", []):
        site = item.get("definition_site")
        if not site:
            continue
        parts = assignment_parts(site.get("code", ""))
        if not parts:
            continue
        lhs, lhs_root, operator, rhs = parts
        role = item.get("origin_role", "")
        if item.get("name") != lhs_root or role not in seed_roles.get(lhs_root, set()):
            reject(rejected, "lhs_target_mismatch")
            continue
        if is_self_update(lhs_root, operator, rhs):
            reject(rejected, "self_update")
            continue
        if is_loop_state(role, lhs_root, rhs, site.get("code", ""), constraints):
            reject(rejected, "loop_state")
            continue
        kind, class_score = endpoint_class(rhs)
        if not kind:
            reject(rejected, "unsupported_transform")
            continue
        matched_file = source_match(source_root, site.get("line"), site.get("code", ""), cache)
        if not matched_file:
            reject(rejected, "definition_not_source_visible")
            continue
        candidates.append({
            "origin_role": role,
            "origin_name": lhs_root,
            "target_identifier": lhs_root,
            "endpoint_class": kind,
            "confidence": "static_candidate",
            "total_hops": item.get("depth"),
            "path_hop": 0,
            "definition_hop": item.get("depth"),
            "caller": row.get("function"),
            "call_line": row.get("line"),
            "argument_code": lhs_root,
            "line": site.get("line"),
            "code": site.get("code"),
            "source_file": matched_file,
            "score": class_score,
        })
    return candidates, rejected


def select_top(candidates: list[dict]):
    grouped = collections.defaultdict(list)
    seen = set()
    for item in candidates:
        key = (item["origin_role"], item["origin_name"], item["line"], item["code"], item["call_line"])
        if key in seen:
            continue
        seen.add(key)
        grouped[(item["origin_role"], item["origin_name"])].append(item)
    result = []
    for group in grouped.values():
        best_score = max(item["score"] for item in group)
        scored = [item for item in group if item["score"] == best_score]
        best_hops = min(item["total_hops"] for item in scored)
        result.extend(sorted(
            (item for item in scored if item["total_hops"] == best_hops),
            key=lambda item: (item["line"], item["code"]),
        )[:2])
    return result


def argument_boundaries(row: dict):
    grouped = collections.defaultdict(list)
    if row.get("argument_path_scope") != "saved_stack_only" or row.get("stack_caller_frames") is None:
        return []
    for path in row.get("argument_paths", []):
        grouped[(path.get("origin_role"), path.get("origin_name"))].append(path)
    result = []
    for group in grouped.values():
        max_hop = max(item.get("hop", 0) for item in group)
        for item in group:
            if item.get("hop") == max_hop:
                result.append({
                    "origin_role": item.get("origin_role"),
                    "origin_name": item.get("origin_name"),
                    "hop": item.get("hop"),
                    "caller": item.get("caller"),
                    "call_line": item.get("call_line"),
                    "argument_code": item.get("argument_code"),
                    "confidence": "exact_stack_argument_boundary",
                })
    return result


def select_row(row: dict, source_root: Path):
    if row.get("status") != "candidate_context":
        return {
            "id": row.get("id"), "status": row.get("status"),
            "selected_definition_endpoints": [], "argument_boundary_candidates": [],
            "constraint_context": [], "uaf_side_effect_candidates": [],
        }
    constraints = []
    seen_constraints = set()
    for item in [*row.get("enclosing_conditions", []), *row.get("control_dependencies", [])]:
        key = (item.get("line"), item.get("code"))
        if key not in seen_constraints and item.get("code"):
            seen_constraints.add(key)
            constraints.append({"line": item.get("line"), "code": item.get("code"), "status": "constraint_context"})
    raw_candidates, rejected = definition_candidates(row, source_root, constraints)
    selected = select_top(raw_candidates)
    uaf = "use-after-free" in row.get("asan_type", "").lower()
    side_effects = []
    if uaf:
        valid = [item for item in row.get("pre_fault_side_effect_candidates", []) if item.get("line", 10**9) < row.get("line", -1)]
        if valid:
            nearest = max(item.get("line", -1) for item in valid)
            side_effects = [
                {**item, "claim": "static_candidate_not_dynamic_free_proof"}
                for item in valid if item.get("line") == nearest
            ]
    return {
        "schema_version": "d-endpoint-selection-v1",
        "id": row.get("id"),
        "status": row.get("status"),
        "max_causal_depth": row.get("max_causal_depth"),
        "selected_definition_endpoints": selected,
        "argument_boundary_candidates": argument_boundaries(row),
        "constraint_context": constraints,
        "uaf_side_effect_candidates": side_effects,
        "endpoint_status": (
            "selected_static_definition" if selected
            else "uaf_static_candidate_only" if side_effects
            else "argument_boundary_only" if argument_boundaries(row)
            else "no_static_endpoint"
        ),
        "excluded_definition_candidate_counts": dict(sorted(rejected.items())),
        "claims": {
            "runtime_path_proven": False,
            "root_cause_proven": False,
            "alias_identity_proven": False,
            "uaf_base_is_freed_object": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [select_row(row, args.source) for row in read_jsonl(args.input)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "definition_endpoints": sum(len(row["selected_definition_endpoints"]) for row in rows),
        "uaf_side_effect_candidates": sum(len(row["uaf_side_effect_candidates"]) for row in rows),
    }))


if __name__ == "__main__":
    main()

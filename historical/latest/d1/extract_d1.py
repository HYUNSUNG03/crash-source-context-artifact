"""Extract a bounded D1 context from a saved Joern graph.

The output separates static candidates from facts. It never claims a runtime
path, root cause, alias identity, or UAF lifetime proof.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Iterable


ACCESS_NAMES = {
    "<operator>.indexAccess": "index_access",
    "<operator>.indirectIndexAccess": "index_access",
    "<operator>.indirection": "dereference",
    "<operator>.fieldAccess": "field_access",
    "<operator>.indirectFieldAccess": "field_access",
}
MEMORY_CALL_SPECS = {
    "memcpy": {"roles": ["destination", "source", "size"], "semantics": "copy"},
    "memmove": {"roles": ["destination", "source", "size"], "semantics": "copy"},
    "memset": {"roles": ["destination", "value", "size"], "semantics": "write"},
    "strcpy": {"roles": ["destination", "source"], "semantics": "copy"},
    "strncpy": {"roles": ["destination", "source", "size"], "semantics": "copy"},
    "fread": {"roles": ["destination", "element_size", "count", "stream"], "semantics": "file_read"},
    "fwrite": {"roles": ["source", "element_size", "count", "stream"], "semantics": "file_write"},
    "FT_NEXT_ULONG": {"roles": ["base"], "semantics": "cursor_read_macro"},
    "FT_NEXT_USHORT": {"roles": ["base"], "semantics": "cursor_read_macro"},
}
MEMORY_CALL_ALIASES = {
    "_TIFFmemcpy": "memcpy",
    "_TIFFmemmove": "memmove",
    "_TIFFmemset": "memset",
}
ALLOCATORS = {"malloc", "calloc", "realloc", "strdup", "operator new"}
DEALLOCATORS = {"free", "operator delete"}


def load(path: Path):
    # Joern's source-code strings may retain literal control characters.
    return json.loads(path.read_text(encoding="utf-8"), strict=False)


def edge_maps(graph, label):
    forward, backward = collections.defaultdict(list), collections.defaultdict(list)
    for edge in graph["edges"].get(label, []):
        forward[edge["src"]].append(edge["dst"])
        backward[edge["dst"]].append(edge["src"])
    return forward, backward


def descendants(root: int, children) -> list[int]:
    result, stack = [], list(children.get(root, []))
    while stack:
        current = stack.pop()
        result.append(current)
        stack.extend(children.get(current, []))
    return result


def ancestors(node_id: int, parents) -> list[int]:
    result, stack, seen = [], list(parents.get(node_id, [])), set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        result.append(current)
        stack.extend(parents.get(current, []))
    return result


def enclosing_condition_rows(node_id, parents, children, nodes):
    rows = []
    for ancestor in ancestors(node_id, parents):
        control = nodes.get(ancestor, {})
        if control.get("label") != "CONTROL_STRUCTURE":
            continue
        expressions = [nodes[x] for x in children.get(ancestor, [])
                       if x in nodes and nodes[x].get("label") != "BLOCK"]
        if not expressions:
            continue
        code = control.get("code", "").lstrip()
        if code.startswith("for"):
            condition = next((row for row in expressions if row.get("order") == 2), expressions[0])
        elif code.startswith("do"):
            condition = max(expressions, key=lambda row: row.get("order", -1))
        else:
            condition = min(expressions, key=lambda row: row.get("order", -1))
        rows.append({"control_node_id": ancestor, "condition_node_id": condition["id"],
                     "line": condition.get("line"), "code": condition.get("code")})
    return rows


def normalized_call(name: str) -> str:
    short = name.rsplit(".", 1)[-1]
    short = re.sub(r"<duplicate>\d+$", "", short)
    return MEMORY_CALL_ALIASES.get(short, short)


def access_candidates(method_ids, line, column, nodes, children, parents):
    candidates = []
    for node_id in method_ids:
        node = nodes[node_id]
        if node.get("line") != line or node.get("label") != "CALL":
            continue
        name = node.get("name", "")
        kind = ACCESS_NAMES.get(name)
        roles, semantics = None, None
        short_name = normalized_call(name)
        if short_name in MEMORY_CALL_SPECS:
            spec = MEMORY_CALL_SPECS[short_name]
            kind, roles, semantics = "memory_call", spec["roles"], spec["semantics"]
        if not kind:
            continue
        access_ancestors = [nodes[x] for x in ancestors(node_id, parents) if x in nodes and nodes[x].get("name") in ACCESS_NAMES]
        outermost = not access_ancestors
        priority = {"memory_call": 5, "index_access": 4, "dereference": 3, "field_access": 2}[kind]
        node_column = node.get("column")
        column_score = -abs(node_column - column) if column is not None and node_column not in {None, -1} else 0
        candidates.append({
            "node_id": node_id,
            "kind": kind,
            "callee_name": name,
            "code": node.get("code", ""),
            "line": line,
            "column": node.get("column"),
            "outermost": outermost,
            "role_names": roles,
            "memory_semantics": semantics,
            "rank_key": [column_score, 1 if outermost else 0, priority, len(node.get("code", ""))],
        })
    candidates.sort(key=lambda row: tuple(row["rank_key"]), reverse=True)
    return candidates[:3]


def source_visible_candidate(candidate, source_line):
    """Require the CPG access to be visible in the retained source line.

    Joern can expose an access produced by macro expansion even though the source
    line contains only the macro invocation.  Such a node is useful internally,
    but it cannot support a source-level operand claim without preprocessing
    provenance.  This conservative gate makes the extractor abstain instead.
    """
    if source_line is None:
        return True
    code = candidate.get("code", "")
    compact_code = re.sub(r"\s+", "", code).rstrip(";")
    compact_source = re.sub(r"\s+", "", source_line)
    if compact_code and compact_code in compact_source:
        return True
    if candidate["kind"] == "memory_call":
        callee = normalized_call(candidate.get("callee_name", ""))
        return bool(callee and re.search(rf"\b{re.escape(callee)}\s*\(", source_line))
    return False


def direct_arguments(node_id, children, nodes):
    values = [nodes[x] for x in children.get(node_id, []) if x in nodes]
    values = [x for x in values if x.get("argument_index", -1) > 0]
    return sorted(values, key=lambda x: (x.get("argument_index", -1), x.get("order", -1)))


def operand_rows(candidate, children, nodes):
    args = direct_arguments(candidate["node_id"], children, nodes)
    if candidate["kind"] == "memory_call":
        roles = candidate["role_names"]
    elif candidate["kind"] == "index_access":
        roles = ["base", "index"]
    elif candidate["kind"] == "dereference":
        roles = ["base"]
    else:
        roles = ["base", "field"]
    return [
        {"role": roles[index] if index < len(roles) else f"argument_{index + 1}",
         "node_id": arg["id"], "code": arg.get("code", ""), "line": arg.get("line")}
        for index, arg in enumerate(args)
    ]


def identifiers_below(root_id, children, nodes, role):
    ids = [root_id, *descendants(root_id, children)]
    result = []
    for node_id in ids:
        node = nodes.get(node_id)
        if not node or node.get("label") != "IDENTIFIER":
            continue
        if not any(existing["name"] == node.get("name") for existing in result):
            result.append({"node_id": node_id, "name": node.get("name", ""),
                           "code": node.get("code", ""), "origin_role": role})
    return result


def causal_operands(candidate, operands, asan_type, children, nodes):
    """Choose memory-semantic operands, not arbitrary call arguments."""
    error = asan_type.lower()
    if candidate["kind"] == "index_access":
        wanted, rationale = {"base", "index"}, "address_is_base_plus_index_including_uaf_underflow"
    elif candidate["kind"] in {"dereference", "field_access"}:
        wanted = {"base"} if "use-after-free" in error else {"base", "index"}
        rationale = "uaf_tracks_dereferenced_object_base" if "use-after-free" in error else "address_is_base_plus_offset"
    elif candidate["kind"] == "memory_call":
        semantics = candidate.get("memory_semantics")
        if semantics == "file_read":
            wanted, rationale = {"destination", "element_size", "count"}, "fread_writes_destination_range"
        elif semantics == "cursor_read_macro":
            wanted, rationale = {"base"}, "macro_reads_and_advances_cursor_base"
        elif "overlap" in error:
            wanted, rationale = {"destination", "source", "size"}, "overlap_requires_both_ranges"
        elif "read" in error:
            wanted, rationale = {"source", "size"}, "asan_read_prioritizes_source_range"
        elif "write" in error:
            wanted, rationale = {"destination", "size"}, "asan_write_prioritizes_destination_range"
        else:
            wanted, rationale = {item["role"] for item in operands}, "access_direction_unavailable"
    else:
        wanted, rationale = {item["role"] for item in operands}, "all_memory_operands"
    selected = [item | {"selection_reason": rationale} for item in operands if item["role"] in wanted]
    if candidate["kind"] == "field_access" and "use-after-free" not in error:
        nested_indexes = [
            nodes[node_id] for node_id in descendants(candidate["node_id"], children)
            if node_id in nodes and nodes[node_id].get("name") in {
                "<operator>.indexAccess", "<operator>.indirectIndexAccess"
            }
        ]
        for nested in nested_indexes:
            args = direct_arguments(nested["id"], children, nodes)
            if len(args) >= 2:
                selected.append({
                    "role": "index",
                    "node_id": args[1]["id"],
                    "code": args[1].get("code", ""),
                    "line": args[1].get("line"),
                    "selection_reason": "nested_index_contributes_to_address",
                })
    seeds = []
    for operand in selected:
        seeds.extend(identifiers_below(operand["node_id"], children, nodes, operand["role"]))
    unique = []
    for seed in seeds:
        if not any((item["node_id"], item["origin_role"]) == (seed["node_id"], seed["origin_role"]) for item in unique):
            unique.append(seed)
    return selected, unique


def definition_site(node_id, parents, nodes):
    definition_names = {"<operator>.assignment", "<operator>.assignmentPlus",
                        "<operator>.assignmentMinus", "<operator>.postIncrement",
                        "<operator>.preIncrement", "<operator>.postDecrement",
                        "<operator>.preDecrement"}
    for ancestor in ancestors(node_id, parents):
        node = nodes.get(ancestor, {})
        if node.get("name") in definition_names:
            return {"node_id": ancestor, "line": node.get("line"), "code": node.get("code")}
    return None


def reaching_closure(seeds, back_rd, nodes, parents, max_depth, focus_line=None, rejected=None):
    rows = []
    for seed in seeds:
        seen, frontier = {seed["node_id"]}, [seed["node_id"]]
        for depth in range(1, max_depth + 1):
            next_frontier = []
            for node_id in frontier:
                for source in back_rd.get(node_id, []):
                    if source in seen:
                        continue
                    node = nodes.get(source, {})
                    source_line = node.get("line", -1)
                    site = definition_site(source, parents, nodes)
                    site_line = (site or {}).get("line", source_line)
                    after_boundary = focus_line is not None and max(source_line, site_line or -1) > focus_line
                    same_statement_definition = (
                        focus_line is not None and site is not None and site_line == focus_line
                    )
                    if after_boundary or same_statement_definition:
                        if rejected is not None:
                            rejected.append({
                                "origin_role": seed["origin_role"],
                                "origin_name": seed["name"],
                                "depth": depth,
                                "node_id": source,
                                "line": source_line,
                                "code": node.get("code"),
                                "definition_site": site,
                                "reason": (
                                    "definition_contains_faulting_statement"
                                    if same_statement_definition else "source_after_fault_or_callsite"
                                ),
                            })
                        seen.add(source)
                        continue
                    seen.add(source)
                    next_frontier.append(source)
                    if node.get("line", -1) > 0 and node.get("label") != "METHOD":
                        rows.append({"origin_role": seed["origin_role"], "origin_name": seed["name"],
                                     "depth": depth, "node_id": source, "line": node.get("line"),
                                     "label": node.get("label"), "name": node.get("name"),
                                     "code": node.get("code"),
                                     "definition_site": site})
            frontier = next_frontier
            if not frontier:
                break
    return rows


def enclosing_method(node_id, methods):
    return next((method for method in methods if node_id in method["node_ids"]), None)


def parameter_index(seed_id, forward_ref, nodes):
    for target in forward_ref.get(seed_id, []):
        node = nodes.get(target, {})
        if node.get("label") == "METHOD_PARAMETER_IN":
            return node.get("order"), node.get("name")
    return None, None


def argument_paths(
    seeds, focus_method, methods, nodes, children, forward_ref, max_depth,
    stack_callers=None, stack_caller_frames=None,
):
    methods_by_name = collections.defaultdict(list)
    for method in methods:
        methods_by_name[method["name"]].append(method)
    paths, frontier = [], [(focus_method, seed, 0) for seed in seeds]
    seen = set()
    while frontier:
        method, seed, depth = frontier.pop(0)
        key = method["id"], seed["node_id"], depth
        if key in seen or depth >= max_depth:
            continue
        seen.add(key)
        index, parameter = parameter_index(seed["node_id"], forward_ref, nodes)
        if not index:
            continue
        for caller in methods:
            if caller["name"] == "<global>":
                continue
            if stack_callers is not None:
                if depth >= len(stack_callers) or caller["name"] != stack_callers[depth]:
                    continue
            expected_line = None
            if stack_caller_frames is not None:
                if depth >= len(stack_caller_frames):
                    continue
                frame = stack_caller_frames[depth]
                if caller["name"] != frame["function"]:
                    continue
                expected_line = frame.get("line")
            for node_id in caller["node_ids"]:
                call = nodes.get(node_id, {})
                if call.get("label") != "CALL" or normalized_call(call.get("name", "")) != normalized_call(method["name"]):
                    continue
                if expected_line is not None and call.get("line") != expected_line:
                    continue
                args = direct_arguments(node_id, children, nodes)
                arg = next((value for value in args if value.get("argument_index") == index), None)
                if not arg:
                    continue
                row = {"hop": depth + 1, "callee": method["name"], "parameter": parameter,
                       "origin_role": seed.get("origin_role"),
                       "origin_name": seed.get("name"),
                       "parameter_index": index, "caller": caller["name"],
                       "call_line": call.get("line"), "call_code": call.get("code"),
                       "argument_code": arg.get("code"), "argument_node_id": arg["id"]}
                paths.append(row)
                for desc_id in [arg["id"], *descendants(arg["id"], children)]:
                    desc = nodes.get(desc_id, {})
                    if desc.get("label") == "IDENTIFIER":
                        frontier.append((caller, {"node_id": desc_id, "name": desc.get("name"),
                                                  "code": desc.get("code"), "origin_role": seed.get("origin_role")}, depth + 1))
    return paths


def enrich_argument_definitions(paths, back_rd, nodes, parents, max_causal_depth):
    enriched = []
    for path in paths:
        seed = {"node_id": path["argument_node_id"], "name": path["argument_code"],
                "code": path["argument_code"], "origin_role": path.get("origin_role")}
        remaining_depth = max(0, max_causal_depth - path["hop"])
        definitions = reaching_closure(
            [seed], back_rd, nodes, parents, remaining_depth,
            focus_line=path.get("call_line"),
        ) if remaining_depth else []
        enriched.append(path | {"caller_reaching_definitions": definitions})
    return enriched


def unique_definition_sites(rows):
    """Collapse CPG node rows to human-reviewable assignment/update sites."""
    result = []
    seen = set()
    for row in rows:
        site = row.get("definition_site")
        if not site:
            continue
        key = (row.get("origin_role"), site.get("node_id"), site.get("line"), site.get("code"))
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "origin_role": row.get("origin_role"),
            "origin_name": row.get("origin_name"),
            "depth": row.get("depth"),
            **site,
        })
    return result


def pre_fault_side_effect_candidates(method, focus_line, seeds, nodes, children):
    """Return prior same-symbol calls as candidates, never as proven effects."""
    names = {seed["name"] for seed in seeds}
    rows = []
    for node_id in method["node_ids"]:
        node = nodes.get(node_id, {})
        line = node.get("line", -1)
        if node.get("label") != "CALL" or line < 0 or line >= focus_line:
            continue
        raw_name = node.get("name", "")
        short = normalized_call(raw_name)
        if raw_name.startswith("<operator>") or raw_name in ACCESS_NAMES:
            continue
        subtree = [nodes[x] for x in [node_id, *descendants(node_id, children)] if x in nodes]
        mentioned = sorted({
            item.get("name") for item in subtree
            if item.get("label") == "IDENTIFIER" and item.get("name") in names
        })
        if not mentioned:
            continue
        rows.append({
            "node_id": node_id,
            "line": line,
            "call": short,
            "code": node.get("code"),
            "matched_seed_names": mentioned,
            "status": "pre_fault_same_symbol_side_effect_candidate_not_proof",
        })
    return sorted(rows, key=lambda row: (row["line"], row["node_id"]))


def lifetime_candidates(method, focus_line, seeds, nodes, children, parents):
    names = {seed["name"] for seed in seeds}
    rows = []
    for node_id in method["node_ids"]:
        node = nodes.get(node_id, {})
        if node.get("label") != "CALL" or node.get("line", -1) > focus_line:
            continue
        short = normalized_call(node.get("name", ""))
        if short not in ALLOCATORS | DEALLOCATORS:
            continue
        scope_ids = [node_id, *descendants(node_id, children)]
        if short in ALLOCATORS:
            assignment = next((ancestor for ancestor in ancestors(node_id, parents)
                               if nodes.get(ancestor, {}).get("name") == "<operator>.assignment"), None)
            if assignment is not None:
                scope_ids.extend([assignment, *descendants(assignment, children)])
        subtree = [nodes[x] for x in scope_ids if x in nodes]
        mentioned = sorted({x.get("name") for x in subtree if x.get("label") == "IDENTIFIER" and x.get("name") in names})
        if mentioned:
            rows.append({"kind": "allocation" if short in ALLOCATORS else "deallocation",
                         "call": short, "line": node.get("line"), "code": node.get("code"),
                         "matched_seed_names": mentioned, "status": "same_symbol_candidate"})
    return rows


def resolve_method(methods, nodes, request_file, requested_name, focus_line):
    """Resolve C++ stack spellings without relaxing file or line identity.

    Saved ASan frames use names such as ``Class::method(args)`` while Joern's
    C/C++ frontend stores ``method`` in Method.name.  Exact name matching stays
    the first choice.  The fallback removes the class/signature from the saved
    spelling and then requires the method AST to contain the exact fault line.
    """
    same_file = [
        method for method in methods
        if not request_file
        or method.get("file", "").replace("\\", "/").endswith(request_file.replace("\\", "/"))
    ]
    exact = [method for method in same_file if method["name"] == requested_name]
    if len(exact) == 1:
        return exact, "exact_saved_symbol"
    if exact:
        return exact, "ambiguous_exact_saved_symbol"
    short = (requested_name or "").split("(", 1)[0].rsplit("::", 1)[-1].strip()
    candidates = []
    for method in same_file:
        if method.get("name") != short:
            continue
        method_lines = {
            nodes[node_id].get("line") for node_id in method.get("node_ids", [])
            if node_id in nodes and nodes[node_id].get("line", -1) > 0
        }
        if focus_line in method_lines:
            candidates.append(method)
    return candidates, "qualified_symbol_basename_plus_exact_fault_line"


def extract(graph, source_root: Path, request, max_rd_depth):
    nodes = {node["id"]: node for node in graph["nodes"]}
    children, parents = edge_maps(graph, "AST")
    forward_ref, _ = edge_maps(graph, "REF")
    _, back_rd = edge_maps(graph, "REACHING_DEF")
    _, back_cdg = edge_maps(graph, "CDG")
    request_file = request.get("file")
    source = source_root / request_file if request_file and source_root.is_dir() else source_root
    source_lines = source.read_text(encoding="utf-8").splitlines() if source.is_file() else []
    if "line" in request:
        line = request["line"]
    else:
        hits = [index + 1 for index, text in enumerate(source_lines) if request["marker"] in text]
        if len(hits) != 1:
            raise ValueError(f"marker {request['marker']} occurs {len(hits)} times")
        line = hits[0]
    method_matches, method_resolution = resolve_method(
        graph["methods"], nodes, request_file, request["function"], line
    )
    if len(method_matches) != 1:
        return {"id": request["id"], "status": "method_resolution_failed", "line": line,
                "method_matches": [{"name": row["name"], "file": row.get("file")} for row in method_matches]}
    method = method_matches[0]
    candidates = access_candidates(method["node_ids"], line, request.get("column"), nodes, children, parents)
    source_line = source_lines[line - 1] if 0 < line <= len(source_lines) else None
    rejected_cpg_only = [row for row in candidates if not source_visible_candidate(row, source_line)]
    candidates = [row for row in candidates if source_visible_candidate(row, source_line)]
    if not candidates:
        return {"id": request["id"], "status": "no_faulting_expression_candidate", "line": line,
                "source_line": source_line,
                "rejected_cpg_only_candidates": [
                    {key: value for key, value in row.items() if key != "rank_key"}
                    for row in rejected_cpg_only
                ]}
    core = candidates[0]
    operands = operand_rows(core, children, nodes)
    selected_operands, seeds = causal_operands(core, operands, request["asan_type"], children, nodes)
    enclosing_conditions = enclosing_condition_rows(core["node_id"], parents, children, nodes)
    control_seeds = []
    if "use-after-free" not in request["asan_type"].lower():
        for condition in enclosing_conditions:
            control_seeds.extend(
                identifiers_below(condition["condition_node_id"], children, nodes, "control")
            )
    all_seeds = []
    for seed in [*seeds, *control_seeds]:
        key = seed["node_id"], seed["origin_role"]
        if not any((item["node_id"], item["origin_role"]) == key for item in all_seeds):
            all_seeds.append(seed)
    rejected_reaching = []
    reaching = reaching_closure(
        all_seeds, back_rd, nodes, parents, max_rd_depth,
        focus_line=line, rejected=rejected_reaching,
    )
    control_subjects = [core["node_id"], *ancestors(core["node_id"], parents)]
    control_ids = list(dict.fromkeys(control for subject in control_subjects
                                     for control in back_cdg.get(subject, [])))
    controls = [{"line": nodes[x].get("line"), "code": nodes[x].get("code"), "node_id": x}
                for x in control_ids if x in nodes]
    stack_callers = request.get("stack_callers")
    stack_caller_frames = request.get("stack_caller_frames")
    call_depth = min(request.get("call_depth", max_rd_depth), max_rd_depth)
    arg_paths = argument_paths(
        all_seeds, method, graph["methods"], nodes, children, forward_ref,
        call_depth, stack_callers, stack_caller_frames,
    )
    arg_paths = enrich_argument_definitions(arg_paths, back_rd, nodes, parents, max_rd_depth)
    lifetime = lifetime_candidates(method, line, seeds, nodes, children, parents)
    side_effects = (
        pre_fault_side_effect_candidates(method, line, seeds, nodes, children)
        if "use-after-free" in request["asan_type"].lower() else []
    )
    return {
        "schema_version": "d-causal-pilot-v2",
        "id": request["id"], "status": "candidate_context", "function": method["name"],
        "saved_function": request["function"], "method_resolution": method_resolution,
        "source_file": request_file or method.get("file"),
        "line": line, "column": request.get("column"), "asan_type": request["asan_type"],
        "faulting_expression_top1": {key: value for key, value in core.items() if key != "rank_key"},
        "faulting_expression_top3": [{key: value for key, value in row.items() if key != "rank_key"} for row in candidates],
        "operands": operands, "causal_operand_candidates": selected_operands,
        "seed_identifiers": seeds,
        "control_seed_identifiers": control_seeds,
        "reaching_definitions": reaching,
        "causal_definition_sites": unique_definition_sites(reaching),
        "rejected_reaching_definitions": rejected_reaching,
        "max_reaching_def_depth": max_rd_depth,
        "control_dependencies": controls,
        "enclosing_conditions": enclosing_conditions,
        "argument_paths": arg_paths, "max_call_depth": call_depth,
        "max_causal_depth": max_rd_depth,
        "causal_depth_semantics": "local_reaching_def_hops_or_stack_argument_hops_plus_caller_rd_hops",
        "argument_path_scope": (
            "saved_stack_only"
            if stack_callers is not None or stack_caller_frames is not None
            else "all_static_callers"
        ),
        "stack_callers": stack_callers,
        "stack_caller_frames": stack_caller_frames,
        "lifetime_event_candidates": lifetime,
        "pre_fault_side_effect_candidates": side_effects,
        "lifetime_status": "candidate_not_path_proof" if lifetime else "not_observed",
        "claims": {"runtime_path_proven": False, "root_cause_proven": False,
                   "alias_identity_proven": False, "faulting_expression_manually_verified": False},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rd-depth", type=int, default=8)
    parser.add_argument("--lifetime-evidence", type=Path)
    args = parser.parse_args()
    graph, requests = load(args.graph), load(args.requests)
    rows = [extract(graph, args.source, request, args.max_rd_depth) for request in requests]
    if args.lifetime_evidence:
        evidence = load(args.lifetime_evidence)
        for row in rows:
            if row.get("id") != evidence.get("report_id"):
                continue
            row["lifetime_evidence"] = evidence
            row["lifetime_status"] = evidence["lifetime_status"]
            row["claims"].update(evidence.get("claims", {}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "available": sum(row["status"] == "candidate_context" for row in rows)}))


if __name__ == "__main__":
    main()

"""Fixed-D4/L numeric experiment with a label-blind freeze boundary.

Run phase1 with the supplied offline Python 3.12 environment first.  It reads
only the frozen protocol and label-free text inventory, encodes the text, and
freezes HDBSCAN candidates/selections.  Run phase2 afterwards to verify that
freeze before it opens ``d4-panel.jsonl`` for scoring.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

# The supplied runtime is a Python 3.12 site-packages tree.  Put it ahead of
# ambient packages before importing NumPy, HDBSCAN, or the encoder stack.
_RUNTIME_PACKAGES = Path(__file__).resolve().parent / "stage4-runtime" / "py312"
if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Run with the supplied Python 3.12 runtime, got {sys.version}")
if str(_RUNTIME_PACKAGES) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_PACKAGES))

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
INPUT_DIR = WORKSPACE / "outputs" / "stage4-fixed-d4-l-inputs"
RESULT_DIR = WORKSPACE / "outputs" / "stage4-fixed-d4-l-results"
RUNTIME_DIR = WORKSPACE / "work" / "stage4-runtime"
MODEL_DIR = RUNTIME_DIR / "models" / "bge-small-en-v1.5-5c38ec7"
LABELS_PATH = WORKSPACE / "outputs" / "stage2-d4-v1" / "d4-panel.jsonl"
GPTRACE_METRICS = Path(
    r"/RESEARCH_WORKSPACES\2026-09-04\referenced-chatgpt-conversation-this-is-an"
    r"\gptrace-source\src\gptrace\ground_truth_analysis.py"
)

DIMENSION = 384
STEPS = 100
JOBS = 4
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
CONTENT_TOKEN_BUDGET = 510
PRIMARY_CONDITIONS = ("L_f1_W0", "L_f1_W5", "L_f3_W0", "L_f3_W5")
SECONDARY_CONDITIONS = ("L_f1_F", "L_f3_F")
ALL_CONDITIONS = PRIMARY_CONDITIONS + SECONDARY_CONDITIONS
EXPECTED_GPTRACE_LF_SHA256 = "4d67e96912cd4ddbb7e2d2e904a77c6330a0713a415d582743a640f794a08968"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty_jsonl:{path}")
    return rows


def partition_hash(values: list[int]) -> str:
    return sha256_bytes(canonical_json([int(value) for value in values]).encode("utf-8"))


def unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim not in (1, 2):
        raise ValueError("unexpected_vector_rank")
    magnitude = np.linalg.norm(value, axis=-1, keepdims=value.ndim == 2)
    if not np.isfinite(value).all() or np.any(magnitude < 1e-12):
        raise ValueError("nonfinite_or_zero_vector")
    return value / magnitude


def verify_protocol_and_inputs() -> tuple[dict, list[dict], dict[str, str]]:
    """Read the Phase 1 inputs.  This function deliberately has no label input."""
    protocol_path = INPUT_DIR / "protocol.json"
    encoding_path = INPUT_DIR / "encoding-inputs.jsonl"
    texts_path = INPUT_DIR / "unique-texts.jsonl"
    protocol = load_json(protocol_path)
    if protocol.get("schema_version") != "fixed-d4-l-protocol-v1":
        raise ValueError("unexpected_protocol_schema")
    encoder = protocol.get("encoder", {})
    expected_encoder = {
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "dimension": DIMENSION,
        "content_token_budget": CONTENT_TOKEN_BUDGET,
        "max_sequence_length": 512,
    }
    for key, expected in expected_encoder.items():
        if encoder.get(key) != expected:
            raise ValueError(f"protocol_encoder_mismatch:{key}")
    if tuple(protocol.get("L", {}).get("primary_candidates", ())) != PRIMARY_CONDITIONS:
        raise ValueError("protocol_primary_condition_order_mismatch")
    if tuple(protocol.get("L", {}).get("secondary_complete_case_only", ())) != SECONDARY_CONDITIONS:
        raise ValueError("protocol_secondary_condition_order_mismatch")
    frozen = protocol.get("frozen_inputs", {})
    for name, path in (("encoding_inputs", encoding_path), ("unique_texts", texts_path)):
        if frozen.get(name, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"protocol_frozen_input_hash_mismatch:{name}")

    input_rows = load_jsonl(encoding_path)
    if any(row.get("schema_version") != "fixed-d4-l-encoding-input-v1" for row in input_rows):
        raise ValueError("unexpected_encoding_input_schema")
    keys = set()
    expected_text_hashes: set[str] = set()
    for row in input_rows:
        condition = row.get("condition")
        report_id = row.get("report_id")
        target = row.get("target")
        if condition not in ALL_CONDITIONS or not report_id or not target:
            raise ValueError("invalid_encoding_input_identity")
        key = (condition, report_id)
        if key in keys:
            raise ValueError(f"duplicate_condition_report:{condition}:{report_id}")
        keys.add(key)
        components = row.get("component_text_sha256")
        if set(components or ()) != {"trace", "no_args_trace", "asan", "L", "D4"}:
            raise ValueError(f"unexpected_component_set:{condition}:{report_id}")
        expected_text_hashes.update(components.values())
        if tuple(row.get("representations", ())) != ("B", "BL", "BD4", "BLD4"):
            raise ValueError(f"unexpected_representation_set:{condition}:{report_id}")

    texts = load_jsonl(texts_path)
    text_by_hash: dict[str, str] = {}
    for row in texts:
        digest, text = row.get("text_sha256"), row.get("text")
        if not isinstance(digest, str) or not isinstance(text, str) or not text:
            raise ValueError("invalid_unique_text")
        if sha256_bytes(text.encode("utf-8")) != digest:
            raise ValueError(f"unique_text_content_hash_mismatch:{digest}")
        if digest in text_by_hash:
            raise ValueError(f"duplicate_unique_text_hash:{digest}")
        text_by_hash[digest] = text
    if set(text_by_hash) != expected_text_hashes:
        raise ValueError("unique_text_inventory_does_not_match_encoding_inputs")
    return protocol, input_rows, text_by_hash


def verify_model_snapshot() -> dict:
    manifest_path = INPUT_DIR / "model-download-manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("model") != MODEL_NAME or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("model_download_manifest_identity_mismatch")
    if not MODEL_DIR.is_dir():
        raise ValueError(f"missing_pinned_model_dir:{MODEL_DIR}")
    observed = {}
    for entry in manifest.get("files", []):
        relative = entry.get("path")
        path = MODEL_DIR / str(relative)
        if not path.is_file() or path.stat().st_size != entry.get("bytes"):
            raise ValueError(f"pinned_model_file_mismatch:{relative}")
        actual = sha256_file(path)
        if actual != entry.get("sha256"):
            raise ValueError(f"pinned_model_hash_mismatch:{relative}")
        observed[str(relative)] = actual
    required = {"config.json", "model.safetensors", "modules.json", "tokenizer.json"}
    if not required.issubset(observed):
        raise ValueError("model_manifest_missing_required_files")
    return {"manifest": sha256_file(manifest_path), "files": observed}


def chunk_tokens(text: str, tokenizer) -> tuple[list[list[int]], dict]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)
    ids = encoded["input_ids"]
    budget = 512 - int(tokenizer.num_special_tokens_to_add(pair=False))
    if budget != CONTENT_TOKEN_BUDGET:
        raise ValueError(f"unexpected_token_budget:{budget}")
    if not ids:
        raise ValueError("empty_text_has_no_tokens")
    # Frozen Stage-4 protocol: consecutive token chunks, with no boundary
    # preference and no chunk cap.
    chunks = [ids[start:start + budget] for start in range(0, len(ids), budget)]
    if any(len(chunk) > CONTENT_TOKEN_BUDGET or not chunk for chunk in chunks):
        raise ValueError("invalid_token_chunk")
    return chunks, {
        "input_tokens": len(ids), "used_tokens": len(ids), "omitted_tokens": 0,
        "chunks": len(chunks), "max_chunk_tokens": max(map(len, chunks)),
    }


def encode_chunks(chunks: list[list[int]], model, batch_size: int = 8) -> np.ndarray:
    import torch

    vectors = []
    tokenizer = model.tokenizer
    for start in range(0, len(chunks), batch_size):
        examples = [
            tokenizer.prepare_for_model(chunk, add_special_tokens=True, truncation=False,
                                        return_attention_mask=True, verbose=False)
            for chunk in chunks[start:start + batch_size]
        ]
        batch = tokenizer.pad(examples, padding=True, return_tensors="pt", verbose=False)
        if int(batch["input_ids"].shape[1]) > 512:
            raise ValueError("encoded_chunk_exceeds_model_limit")
        with torch.inference_mode():
            values = model(dict(batch))["sentence_embedding"].detach().cpu().numpy()
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not np.isfinite(values).all() or np.any(norms < 1e-12):
            raise ValueError("invalid_chunk_embedding")
        vectors.append(values / norms)
    return unit(np.concatenate(vectors, axis=0).mean(axis=0)).astype(np.float32)


def encode_texts(text_by_hash: dict[str, str]) -> tuple[dict[str, np.ndarray], dict[str, dict], dict]:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(JOBS)
    torch.manual_seed(0)
    print(f"Loading pinned BGE model from {MODEL_DIR}", flush=True)
    model = SentenceTransformer(str(MODEL_DIR), device="cpu", trust_remote_code=False, local_files_only=True)
    model.max_seq_length = 512
    model.eval()
    if model.get_sentence_embedding_dimension() != DIMENSION:
        raise ValueError("unexpected_embedding_dimension")
    parity_text = "void calculate(int count) { int total = count + 1; }"
    parity_chunks, _ = chunk_tokens(parity_text, model.tokenizer)
    parity_own = encode_chunks(parity_chunks, model)
    parity_library = model.encode(parity_text, normalize_embeddings=True, show_progress_bar=False)
    if not np.allclose(parity_own, parity_library, atol=2e-5):
        raise ValueError("custom_short_text_encoding_parity_failed")
    vectors: dict[str, np.ndarray] = {}
    token_counts: dict[str, dict] = {}
    ordered = sorted(text_by_hash.items())
    started = time.monotonic()
    print(f"Encoding {len(ordered)} label-free unique texts", flush=True)
    for number, (digest, text) in enumerate(ordered, 1):
        chunks, counts = chunk_tokens(text, model.tokenizer)
        vector = encode_chunks(chunks, model)
        if vector.shape != (DIMENSION,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5):
            raise ValueError(f"invalid_embedding:{digest}")
        vectors[digest], token_counts[digest] = vector, counts
        if number % 20 == 0 or number == len(ordered):
            print(f"Encoded {number}/{len(ordered)} unique texts in {time.monotonic() - started:.1f}s", flush=True)
    return vectors, token_counts, {
        "model": MODEL_NAME, "revision": MODEL_REVISION, "embedding_dimension": DIMENSION,
        "dtype": "float32", "chunking": "consecutive_token_ids_v1",
        "content_token_budget": CONTENT_TOKEN_BUDGET, "max_sequence_length": 512,
        "batch_size": 8, "threads": JOBS,
        "aggregation": "equal_mean_of_unit_chunk_vectors_then_normalize", "prompt": None,
        "truncated_unique_texts": 0,
        "short_encode_parity_cosine": float(unit(parity_own) @ unit(parity_library)),
    }


def relabel_noise(raw_partition: list[int]) -> list[int]:
    values = [int(value) for value in raw_partition]
    next_noise = min(values + [0]) - 1
    for index, value in enumerate(values):
        if value < 0:
            values[index] = next_noise
            next_noise -= 1
    return values


def fit_candidate(vectors: np.ndarray, epsilon: float, cache_path: Path) -> dict:
    from hdbscan import HDBSCAN

    fitted = HDBSCAN(
        min_cluster_size=2, min_samples=1, metric="euclidean",
        cluster_selection_epsilon=float(epsilon), gen_min_span_tree=True,
        allow_single_cluster=True, cluster_selection_method="eom",
        core_dist_n_jobs=JOBS, memory=str(cache_path),
    ).fit(vectors)
    raw = [int(value) for value in fitted.labels_]
    partition = relabel_noise(raw)
    try:
        dbcv = float(fitted.relative_validity_)
    except (AttributeError, ValueError):
        dbcv = -1.0
    persistence_values = [float(value) for value in fitted.cluster_persistence_]
    persistence = float(np.mean(persistence_values)) if persistence_values else 0.0
    if not math.isfinite(dbcv):
        dbcv = -1.0
    if not math.isfinite(persistence):
        persistence = 0.0
    return {
        "epsilon": float(epsilon), "dbcv": dbcv, "persistence": persistence,
        "cluster_persistence": persistence_values,
        "reported_cluster_count": len(set(partition)),
        "raw_non_noise_structural_cluster_count": len({value for value in raw if value >= 0}),
        "noise_point_count": sum(value < 0 for value in raw),
        "raw_partition_sha256": partition_hash(raw), "partition_sha256": partition_hash(partition),
        "raw_partition": raw, "partition": partition,
    }


def enumerate_sweep(vectors: np.ndarray, cache_path: Path) -> tuple[list[dict], dict]:
    """The frozen 100-step adaptive sweep from cluster_control_pilot_v1.py."""
    from sklearn.metrics import pairwise_distances

    distances = pairwise_distances(vectors, metric="euclidean")
    nonzero = distances[distances > 0]
    if len(nonzero) == 0:
        raw = [0] * len(vectors)
        return [{
            "epsilon": 0.0, "dbcv": -1.0, "persistence": 1.0,
            "cluster_persistence": [1.0], "reported_cluster_count": 1,
            "raw_non_noise_structural_cluster_count": 1, "noise_point_count": 0,
            "raw_partition_sha256": partition_hash(raw), "partition_sha256": partition_hash(raw),
            "raw_partition": raw, "partition": raw, "evaluation_index": 0,
        }], {"distance_min": 0.0, "distance_max": 0.0, "all_vectors_identical": True, "endpoint_visits": 0}
    minimum, maximum = float(nonzero.min()), float(nonzero.max())
    stack: list[tuple[float, float]] = [(minimum, maximum)]
    cached: dict[float, dict] = {}
    endpoint_visits = 0
    for _ in range(STEPS):
        if not stack:
            break
        start, end = stack.pop(0)
        endpoints = []
        for epsilon in (start, end):
            endpoint_visits += 1
            key = round(epsilon, 15)
            if key not in cached:
                item = fit_candidate(vectors, epsilon, cache_path)
                item["evaluation_index"] = len(cached)
                cached[key] = item
            endpoints.append(cached[key])
        left, right = endpoints
        if (left["reported_cluster_count"] != right["reported_cluster_count"]
                or left["dbcv"] != right["dbcv"]):
            middle = (start + end) / 2
            step = (middle - start) / STEPS
            if start + step < middle:
                stack.append((start + step, middle))
            if middle + step < end - step:
                stack.append((middle + step, end - step))
    return list(cached.values()), {
        "distance_min": minimum, "distance_max": maximum,
        "all_vectors_identical": False, "endpoint_visits": endpoint_visits,
    }


# Exact S0/Smax rules frozen from work/l-grid-v1/hdbscan_selector_v2.py.
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


def fuse_rows(rows: list[dict], vector_by_hash: dict[str, np.ndarray]) -> dict[tuple[str, str], dict]:
    result: dict[tuple[str, str], dict] = {}
    for row in rows:
        parts = row["component_text_sha256"]
        trace, no_args, asan, loc, d4 = (unit(vector_by_hash[parts[name]]) for name in ("trace", "no_args_trace", "asan", "L", "D4"))
        base = unit(np.mean([trace, no_args, asan], axis=0))
        values = {"B": base, "BL": unit(base + loc), "BD4": unit(base + d4), "BLD4": unit(base + loc + d4)}
        key = (row["condition"], row["report_id"])
        result[key] = {"target": row["target"], "values": values, "components": parts}
    return result


def candidate_annotation(item: dict) -> dict:
    return {
        "evaluation_index": int(item["evaluation_index"]), "epsilon": float(item["epsilon"]),
        "relative_validity_hdbscan": float(item["dbcv"]), "persistence": float(item["persistence"]),
        "raw_structural_clusters": int(item["raw_non_noise_structural_cluster_count"]),
        "noise_points": int(item["noise_point_count"]), "reported_partition_clusters": int(item["reported_cluster_count"]),
        "raw_partition_sha256": item["raw_partition_sha256"], "partition_sha256": item["partition_sha256"],
        "single_cluster_correction": (int(item["raw_non_noise_structural_cluster_count"]) == 1
                                      and int(item["noise_point_count"]) == 0
                                      and abs(float(item["dbcv"]) - 0.5) <= 1e-12),
    }


def phase1(args) -> None:
    """Make all numeric selections without loading a label-bearing file."""
    started = time.monotonic()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    freeze_path = RESULT_DIR / "phase1-freeze.json"
    if freeze_path.exists():
        raise ValueError("phase1_already_frozen_use_a_new_result_directory")
    protocol, input_rows, text_by_hash = verify_protocol_and_inputs()
    model_provenance = verify_model_snapshot()
    try:
        hdbscan_version = importlib.metadata.version("hdbscan")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("missing_required_hdbscan_0_8_40") from exc
    if hdbscan_version != "0.8.40":
        raise ValueError(f"hdbscan_version_mismatch:{hdbscan_version}")

    vectors, tokens_by_hash, encoder = encode_texts(text_by_hash)
    vector_hashes = sorted(vectors)
    vector_array = np.stack([vectors[digest] for digest in vector_hashes]).astype(np.float32, copy=False)
    vectors_path = RESULT_DIR / "bge-vectors.npz"
    np.savez_compressed(vectors_path, hashes=np.array(vector_hashes), vectors=vector_array)
    encoder_manifest_path = RESULT_DIR / "encoding-manifest.json"
    encoder_manifest = {
        **encoder, "unique_texts": len(vector_hashes), "records": len(input_rows),
        "tokens_by_hash": tokens_by_hash, "input_sha256": sha256_file(INPUT_DIR / "encoding-inputs.jsonl"),
        "unique_texts_sha256": sha256_file(INPUT_DIR / "unique-texts.jsonl"),
        "vectors_sha256": sha256_file(vectors_path),
        "vector_hash_keys_sha256": sha256_bytes("\n".join(vector_hashes).encode("utf-8")),
        "vector_array_float32_sha256": sha256_bytes(np.ascontiguousarray(vector_array).tobytes()),
        "model_snapshot": model_provenance,
        "encoder_script_sha256": sha256_file(Path(__file__)),
    }
    write_json(encoder_manifest_path, encoder_manifest)

    fused = fuse_rows(input_rows, vectors)
    grouped: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for condition, report_id in fused:
        grouped[(condition, fused[(condition, report_id)]["target"])].append(report_id)
    matrix_rows: list[dict] = []
    candidate_rows: list[dict] = []
    selection_rows: list[dict] = []
    computed: dict[str, tuple[list[dict], dict, str]] = {}
    cache_root = RESULT_DIR / "hdbscan-cache"
    for condition in ALL_CONDITIONS:
        for target in sorted(target for candidate_condition, target in grouped if candidate_condition == condition):
            report_ids = sorted(grouped[(condition, target)])
            if len(report_ids) < 2:
                raise ValueError(f"too_few_reports:{condition}:{target}")
            # Build each of the four fixed fusion matrices in declared order.
            for representation in ("B", "BL", "BD4", "BLD4"):
                matrix = np.stack([fused[(condition, report_id)]["values"][representation]
                                   for report_id in report_ids]).astype(np.float64, copy=False)
                matrix_sha256 = sha256_bytes(np.ascontiguousarray(matrix).tobytes())
                key = f"{condition}/{target}/{representation}"
                if matrix_sha256 in computed:
                    items, sweep, reused_from = computed[matrix_sha256]
                    runtime_kind = "reused_matrix_lookup"
                else:
                    print(f"Sweeping {key}: {len(report_ids)} reports", flush=True)
                    items, sweep = enumerate_sweep(matrix, cache_root / matrix_sha256)
                    computed[matrix_sha256] = (items, sweep, key)
                    reused_from = None
                    runtime_kind = "computed_hdbscan_sweep"
                s0, s0_trace = choose_s0(items)
                smax, smax_trace = choose_smax(items)
                if smax is None:
                    raise ValueError(f"smax_unavailable:{key}")
                for item in items:
                    candidate_rows.append({
                        "schema_version": "stage4-fixed-d4-l-candidate-v1", "key": key,
                        "condition": condition, "condition_tier": ("primary_full_coverage" if condition in PRIMARY_CONDITIONS else "secondary_complete_case"),
                        "target": target, "representation": representation, "report_ids_sha256": sha256_bytes("\n".join(report_ids).encode("utf-8")),
                        "matrix_sha256": matrix_sha256, "reused_from": reused_from, **item,
                    })
                selection_rows.extend([
                    {"schema_version": "stage4-fixed-d4-l-selection-v1", "key": key, "selector": "S0",
                     "condition": condition, "target": target, "representation": representation,
                     "matrix_sha256": matrix_sha256, "report_ids": report_ids, "partition": s0["partition"],
                     **candidate_annotation(s0), "selection_trace": s0_trace, "labels_loaded": False},
                    {"schema_version": "stage4-fixed-d4-l-selection-v1", "key": key, "selector": "Smax",
                     "condition": condition, "target": target, "representation": representation,
                     "matrix_sha256": matrix_sha256, "report_ids": report_ids, "partition": smax["partition"],
                     **candidate_annotation(smax), "selection_trace": smax_trace, "labels_loaded": False},
                ])
                matrix_rows.append({
                    "schema_version": "stage4-fixed-d4-l-matrix-v1", "key": key,
                    "condition": condition, "condition_tier": ("primary_full_coverage" if condition in PRIMARY_CONDITIONS else "secondary_complete_case"),
                    "target": target, "representation": representation, "reports": len(report_ids),
                    "report_ids": report_ids, "report_ids_sha256": sha256_bytes("\n".join(report_ids).encode("utf-8")),
                    "matrix_sha256": matrix_sha256, "dimension": DIMENSION, "sweep": sweep,
                    "candidate_count": len(items), "reused_from": reused_from,
                    "clustering_runtime_kind": runtime_kind,
                    "component_hashes": [{"report_id": report_id, **fused[(condition, report_id)]["components"]}
                                         for report_id in report_ids],
                })
                print(f"Frozen {key}: {len(items)} candidates; S0={s0['evaluation_index']}; Smax={smax['evaluation_index']}", flush=True)

    candidates_path = RESULT_DIR / "candidates.jsonl"
    matrices_path = RESULT_DIR / "matrices.jsonl"
    selections_path = RESULT_DIR / "selections.jsonl"
    write_jsonl(candidates_path, candidate_rows)
    write_jsonl(matrices_path, matrix_rows)
    write_jsonl(selections_path, selection_rows)
    coverage = {condition: len({report_id for current_condition, report_id in fused if current_condition == condition})
                for condition in ALL_CONDITIONS}
    if any(coverage[condition] != 25 for condition in PRIMARY_CONDITIONS):
        raise ValueError("primary_conditions_are_not_full_coverage")
    manifest = {
        "schema_version": "stage4-fixed-d4-l-phase1-freeze-v1", "status": "frozen_label_blind",
        "labels_loaded": False, "phase1_cli_has_label_argument": False,
        "protocol_sha256": sha256_file(INPUT_DIR / "protocol.json"),
        "input_files": {path.name: sha256_file(path) for path in (
            INPUT_DIR / "protocol.json", INPUT_DIR / "encoding-inputs.jsonl", INPUT_DIR / "unique-texts.jsonl", INPUT_DIR / "model-download-manifest.json",
        )},
        "model": MODEL_NAME, "revision": MODEL_REVISION, "model_snapshot": model_provenance,
        "hdbscan_version": hdbscan_version, "hdbscan": {"metric": "euclidean", "min_cluster_size": 2,
            "min_samples": 1, "cluster_selection_method": "eom", "allow_single_cluster": True,
            "candidate_sweep": "exact_cluster_control_pilot_v1_100_step_adaptive",
            "candidate_sweep_source": {"path": "/RESEARCH_WORKSPACES/2026-09-05/new-chat-2/work/cluster_control_pilot_v1.py",
                "sha256": sha256_file(Path(r"/RESEARCH_WORKSPACES\2026-09-05\new-chat-2\work\cluster_control_pilot_v1.py"))}},
        "selector_source": {"path": "work/l-grid-v1/hdbscan_selector_v2.py",
                            "sha256": sha256_file(WORKSPACE / "work" / "l-grid-v1" / "hdbscan_selector_v2.py"),
                            "selectors": ["S0", "Smax"]},
        "condition_coverage": coverage, "primary_conditions": list(PRIMARY_CONDITIONS),
        "secondary_conditions_complete_case_only": list(SECONDARY_CONDITIONS),
        "matrix_count": len(matrix_rows), "selection_count": len(selection_rows),
        "candidate_count": len(candidate_rows), "unique_matrix_count": len(computed),
        "artifacts": {path.name: sha256_file(path) for path in (vectors_path, encoder_manifest_path, candidates_path, matrices_path, selections_path)},
        "generator": {"file": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        "freeze_boundary": "all listed input and artifact hashes; freeze manifest does not hash itself",
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(freeze_path, manifest)
    print(json.dumps({"phase": "phase1_complete", "freeze": str(freeze_path), "matrices": len(matrix_rows),
                      "selections": len(selection_rows), "labels_loaded": False}, indent=2), flush=True)


def load_vectors(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        hashes = [str(value) for value in archive["hashes"].tolist()]
        vectors = archive["vectors"]
    if vectors.shape != (len(hashes), DIMENSION) or vectors.dtype != np.dtype("float32"):
        raise ValueError("invalid_frozen_vector_archive")
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate_frozen_vector_hash")
    return {digest: unit(vector) for digest, vector in zip(hashes, vectors)}


def verify_phase1_freeze() -> tuple[dict, list[dict], list[dict], dict[str, np.ndarray], list[dict]]:
    """Verify selection artifacts before the caller may read d4-panel.jsonl."""
    freeze_path = RESULT_DIR / "phase1-freeze.json"
    manifest = load_json(freeze_path)
    if manifest.get("schema_version") != "stage4-fixed-d4-l-phase1-freeze-v1" or manifest.get("labels_loaded") is not False:
        raise ValueError("phase1_not_a_label_blind_freeze")
    for name, expected in manifest["input_files"].items():
        path = INPUT_DIR / name
        if sha256_file(path) != expected:
            raise ValueError(f"phase1_input_hash_mismatch:{name}")
    for name, expected in manifest["artifacts"].items():
        if sha256_file(RESULT_DIR / name) != expected:
            raise ValueError(f"phase1_artifact_hash_mismatch:{name}")
    generator = manifest.get("generator", {})
    if sha256_file(Path(generator.get("file", ""))) != generator.get("sha256"):
        raise ValueError("phase1_generator_changed_after_freeze")
    if manifest.get("hdbscan_version") != "0.8.40":
        raise ValueError("phase1_hdbscan_not_0_8_40")
    matrices = load_jsonl(RESULT_DIR / "matrices.jsonl")
    candidates = load_jsonl(RESULT_DIR / "candidates.jsonl")
    selections = load_jsonl(RESULT_DIR / "selections.jsonl")
    selection_groups: dict[str, list[dict]] = collections.defaultdict(list)
    candidate_groups: dict[str, list[dict]] = collections.defaultdict(list)
    for row in selections:
        selection_groups[row["key"]].append(row)
    for row in candidates:
        candidate_groups[row["key"]].append(row)
    keys = {row["key"] for row in matrices}
    if keys != set(selection_groups) or keys != set(candidate_groups) or len(keys) != len(matrices):
        raise ValueError("phase1_key_sets_mismatch")
    for matrix in matrices:
        key = matrix["key"]
        report_ids = matrix["report_ids"]
        if report_ids != sorted(report_ids) or len(report_ids) != len(set(report_ids)):
            raise ValueError(f"frozen_report_order_invalid:{key}")
        if matrix["report_ids_sha256"] != sha256_bytes("\n".join(report_ids).encode("utf-8")):
            raise ValueError(f"frozen_report_order_hash_mismatch:{key}")
        candidates_for_key = candidate_groups[key]
        if [row["evaluation_index"] for row in candidates_for_key] != list(range(len(candidates_for_key))):
            raise ValueError(f"frozen_candidate_order_invalid:{key}")
        for selected in selection_groups[key]:
            matching = [row for row in candidates_for_key if row["evaluation_index"] == selected["evaluation_index"]]
            if len(matching) != 1 or matching[0]["partition"] != selected["partition"]:
                raise ValueError(f"frozen_selection_partition_mismatch:{key}:{selected['selector']}")
            if partition_hash(selected["partition"]) != selected["partition_sha256"]:
                raise ValueError(f"frozen_selection_hash_mismatch:{key}:{selected['selector']}")
    vector_map = load_vectors(RESULT_DIR / "bge-vectors.npz")
    input_rows = load_jsonl(INPUT_DIR / "encoding-inputs.jsonl")
    return manifest, matrices, selections, vector_map, input_rows


def normalized_lf_bytes(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def load_pinned_metrics():
    raw = GPTRACE_METRICS.read_bytes()
    raw_hash, normalized_hash = sha256_bytes(raw), sha256_bytes(normalized_lf_bytes(raw))
    if normalized_hash != EXPECTED_GPTRACE_LF_SHA256:
        raise ValueError("pinned_gptrace_ground_truth_analysis_normalized_hash_mismatch")
    spec = importlib.util.spec_from_file_location("stage4_gptrace_ground_truth_analysis", GPTRACE_METRICS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, {"path": str(GPTRACE_METRICS), "raw_sha256": raw_hash,
                    "lf_normalized_sha256": normalized_hash}


def error_counts(predicted: list[int], truth: list[str]) -> dict:
    by_bug: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for cluster, bug in zip(predicted, truth):
        by_bug[bug][cluster] += 1
    clusters = {cluster for values in by_bug.values() for cluster in values}
    under = sum(sum(cluster in values for values in by_bug.values()) > 1 for cluster in clusters)
    over = sum(len(values) > 1 for values in by_bug.values())
    lost = sum(all(any(cluster in other for other_bug, other in by_bug.items() if other_bug != bug)
                   for cluster in values) for bug, values in by_bug.items())
    return {"num_overcount": over, "num_undercount": under, "num_completely_lost": lost}


def score_metrics(metric_module, matrix: np.ndarray, predicted: list[int], truth: list[str]) -> dict:
    from sklearn.metrics import adjusted_rand_score, average_precision_score, pairwise_distances

    purity, inverse_purity, f_measure = metric_module.statistical_scores(predicted, truth)
    distances = pairwise_distances(matrix, metric="euclidean")
    pair_truth, pair_scores = [], []
    for left in range(len(truth)):
        for right in range(left + 1, len(truth)):
            pair_truth.append(int(truth[left] == truth[right]))
            pair_scores.append(-float(distances[left, right]))
    pairwise_ap = float(average_precision_score(pair_truth, pair_scores)) if len(set(pair_truth)) == 2 else None
    return {"purity": purity, "inverse_purity": inverse_purity, "f_measure": f_measure,
            "adjusted_rand_index": float(adjusted_rand_score(truth, predicted)),
            "pairwise_average_precision": pairwise_ap, "pairwise_same_bug_pairs": sum(pair_truth),
            "pairwise_different_bug_pairs": len(pair_truth) - sum(pair_truth), **error_counts(predicted, truth)}


def phase2(args) -> None:
    """Verify Phase 1, then and only then join the stage-2 label panel."""
    started = time.monotonic()
    manifest, matrices, selections, vector_map, input_rows = verify_phase1_freeze()
    print("Verified Phase 1 freeze; opening stage-2 labels for scoring", flush=True)

    # This is intentionally the first annotation read in this function.
    label_rows = load_jsonl(LABELS_PATH)
    labels: dict[str, dict] = {}
    for row in label_rows:
        report_id = row.get("report_id")
        if not report_id or report_id in labels or not isinstance(row.get("label"), str):
            raise ValueError("invalid_or_duplicate_stage2_label")
        labels[report_id] = row
    frozen_ids = {report_id for row in input_rows for report_id in [row["report_id"]]}
    if set(labels) != frozen_ids:
        raise ValueError("stage2_labels_do_not_exactly_match_frozen_inventory")
    for row in input_rows:
        if labels[row["report_id"]].get("target") != row["target"]:
            raise ValueError(f"stage2_label_target_mismatch:{row['report_id']}")
    metric_module, metric_provenance = load_pinned_metrics()
    fused = fuse_rows(input_rows, vector_map)
    selection_by_key: dict[str, list[dict]] = collections.defaultdict(list)
    for row in selections:
        selection_by_key[row["key"]].append(row)

    results: list[dict] = []
    for matrix_record in matrices:
        key = matrix_record["key"]
        condition, target, representation = (matrix_record[name] for name in ("condition", "target", "representation"))
        report_ids = matrix_record["report_ids"]
        truth = [labels[report_id]["label"] for report_id in report_ids]
        matrix = np.stack([fused[(condition, report_id)]["values"][representation] for report_id in report_ids]).astype(np.float64, copy=False)
        if sha256_bytes(np.ascontiguousarray(matrix).tobytes()) != matrix_record["matrix_sha256"]:
            raise ValueError(f"frozen_matrix_rebuild_hash_mismatch:{key}")
        for selected in sorted(selection_by_key[key], key=lambda value: value["selector"]):
            predicted = [int(value) for value in selected["partition"]]
            if len(predicted) != len(truth) or partition_hash(predicted) != selected["partition_sha256"]:
                raise ValueError(f"frozen_partition_recheck_failed:{key}:{selected['selector']}")
            results.append({
                "condition": condition, "condition_tier": matrix_record["condition_tier"], "target": target,
                "representation": representation, "selector": selected["selector"], "reports": len(report_ids),
                "true_bugs": len(set(truth)), "predicted_clusters": len(set(predicted)),
                "matrix_sha256": matrix_record["matrix_sha256"], "partition_sha256": selected["partition_sha256"],
                "evaluation_index": selected["evaluation_index"], "epsilon": selected["epsilon"],
                "relative_validity_hdbscan": selected["relative_validity_hdbscan"], "persistence": selected["persistence"],
                **score_metrics(metric_module, matrix, predicted, truth),
            })
        print(f"Scored {key}", flush=True)

    results_path = RESULT_DIR / "scores.csv"
    with results_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summaries = []
    for condition in ALL_CONDITIONS:
        for representation in ("B", "BL", "BD4", "BLD4"):
            for selector in ("S0", "Smax"):
                rows = [row for row in results if row["condition"] == condition and row["representation"] == representation and row["selector"] == selector]
                if not rows:
                    continue
                pairwise_aps = [row["pairwise_average_precision"] for row in rows
                                if row["pairwise_average_precision"] is not None]
                summaries.append({
                    "condition": condition, "condition_tier": rows[0]["condition_tier"], "representation": representation,
                    "selector": selector, "targets": len(rows), "reports": sum(row["reports"] for row in rows),
                    "macro_purity": float(np.mean([row["purity"] for row in rows])),
                    "macro_inverse_purity": float(np.mean([row["inverse_purity"] for row in rows])),
                    "macro_f_measure": float(np.mean([row["f_measure"] for row in rows])),
                    "macro_adjusted_rand_index": float(np.mean([row["adjusted_rand_index"] for row in rows])),
                    "macro_pairwise_average_precision": float(np.mean(pairwise_aps)) if pairwise_aps else None,
                })
    summary_path = RESULT_DIR / "score-summary.json"
    write_json(summary_path, {"primary_conditions": list(PRIMARY_CONDITIONS),
                              "secondary_complete_case_only": list(SECONDARY_CONDITIONS), "rows": summaries})
    phase2_manifest_path = RESULT_DIR / "phase2-manifest.json"
    phase2_manifest = {
        "schema_version": "stage4-fixed-d4-l-phase2-manifest-v1", "status": "scored_after_phase1_verification",
        "phase1_verified_before_label_read": True, "phase1_freeze_sha256": sha256_file(RESULT_DIR / "phase1-freeze.json"),
        "stage2_label_file": str(LABELS_PATH), "stage2_label_sha256": sha256_file(LABELS_PATH),
        "label_inventory_exactly_matches_frozen_reports": True, "gptrace_metric": metric_provenance,
        "metric_names": ["GPTrace purity", "GPTrace inverse purity", "GPTrace F", "sklearn ARI", "pairwise AP"],
        "score_rows": len(results), "outputs": {results_path.name: sha256_file(results_path), summary_path.name: sha256_file(summary_path)},
        "generator_sha256": sha256_file(Path(__file__)), "elapsed_seconds": time.monotonic() - started,
    }
    write_json(phase2_manifest_path, phase2_manifest)
    print(json.dumps({"phase": "phase2_complete", "scores": str(results_path), "rows": len(results)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("phase1", "phase2"))
    args = parser.parse_args()
    if args.phase == "phase1":
        phase1(args)
    else:
        phase2(args)


if __name__ == "__main__":
    main()

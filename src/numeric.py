"""Functions extracted from the frozen experiment, without workstation-specific globals.
Requires NumPy; encoding additionally requires the recorded model runtime.
"""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
DIMENSION=384
CONTENT_TOKEN_BUDGET=510
STEPS=100
JOBS=4

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

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

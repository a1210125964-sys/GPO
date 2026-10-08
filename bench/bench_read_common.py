#!/usr/bin/env python3
"""
Shared dataset configuration and exact-search utilities for the read
experiments.

What it does:
  1. Centrally defines each dataset's dimension, column name, canonical
     distance, opclass, operator, and table name
  2. Provides deterministic-order Parquet vector I/O and file SHA-256
  3. Provides batched exact L2/cosine top-k and Ground Truth CSV I/O

Usage:
    python3 -c "from util.bench_read_common import DATASETS; print(DATASETS)"

Command line:
    This file is a shared module with no command-line arguments; it is
    imported by scripts 01, 03, and 12.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
QUERY_DIR = DATA_DIR / "queries"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dimension: int
    base_column: str
    metric: str
    table: str

    @property
    def operator(self) -> str:
        return "<->" if self.metric == "l2" else "<=>"

    @property
    def operator_class(self) -> str:
        return "vector_l2_ops" if self.metric == "l2" else "vector_cosine_ops"

    @property
    def base_dir(self) -> Path:
        return DATA_DIR / self.name

    @property
    def query_file(self) -> Path:
        return QUERY_DIR / self.name / "query-1000.parquet"

    @property
    def ground_truth_file(self) -> Path:
        return QUERY_DIR / self.name / "ground-truth-k10.csv"


DATASETS: dict[str, DatasetSpec] = {
    "COCOI2I100K": DatasetSpec(
        "COCOI2I100K", 512, "embedding", "cosine", "bench_cocoi2i100k"
    ),
    "Deep100K": DatasetSpec(
        "Deep100K", 96, "embedding", "l2", "bench_deep100k"
    ),
    "Glove100K": DatasetSpec(
        "Glove100K", 200, "openai", "cosine", "bench_glove100k"
    ),
    "NYTimes100K": DatasetSpec(
        "NYTimes100K", 256, "embedding", "cosine", "bench_nytimes100k"
    ),
    "SIFT100K": DatasetSpec(
        "SIFT100K", 128, "embedding", "l2", "bench_sift100k"
    ),
    "Text2image100K": DatasetSpec(
        "Text2image100K", 200, "embedding", "cosine", "bench_text2image100k"
    ),
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_files(path: Path) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found at {path}")
    return files


def load_parquet_vectors(path: Path, column: str, dimension: int) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for parquet_path in parquet_files(path):
        table = pq.read_table(parquet_path, columns=[column])
        values = table.column(column)
        array = np.asarray(values.to_pylist(), dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != dimension:
            raise ValueError(
                f"{parquet_path}: expected (*, {dimension}), got {array.shape}"
            )
        arrays.append(array)
    return np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)


def write_parquet_vectors(path: Path, vectors: np.ndarray) -> None:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected a matrix, got {vectors.shape}")
    flattened = pa.array(vectors.reshape(-1), type=pa.float32())
    embeddings = pa.FixedSizeListArray.from_arrays(
        flattened, list_size=vectors.shape[1]
    )
    pq.write_table(
        pa.Table.from_arrays([embeddings], names=["embedding"]),
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
    )


def exact_topk(
    base: np.ndarray,
    queries: np.ndarray,
    metric: str,
    k: int,
    batch_size: int = 20,
) -> np.ndarray:
    """Return 1-based base-row IDs for exact top-k neighbors."""
    if metric not in {"l2", "cosine"}:
        raise ValueError(f"Unsupported metric: {metric}")
    if len(base) < k:
        raise ValueError(f"Base has only {len(base)} rows, fewer than k={k}")

    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    base_ids = np.arange(1, len(base) + 1, dtype=np.int64)

    if metric == "cosine":
        base_norms = np.linalg.norm(base, axis=1)
        valid_base = base_norms > 0
        normalized_base = np.zeros_like(base)
        normalized_base[valid_base] = base[valid_base] / base_norms[valid_base, None]
        query_norms = np.linalg.norm(queries, axis=1)
        if np.any(query_norms == 0):
            raise ValueError("Cosine query set contains zero vectors")
    else:
        base_squared = np.einsum("ij,ij->i", base, base)

    result = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        query_batch = queries[start:end]
        if metric == "cosine":
            normalized_queries = query_batch / query_norms[start:end, None]
            distances = 1.0 - normalized_queries @ normalized_base.T
            distances[:, ~valid_base] = np.inf
        else:
            query_squared = np.einsum("ij,ij->i", query_batch, query_batch)
            distances = (
                query_squared[:, None]
                + base_squared[None, :]
                - 2.0 * query_batch @ base.T
            )

        candidates = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        for row_index, candidate_indices in enumerate(candidates):
            candidate_distances = distances[row_index, candidate_indices]
            order = np.lexsort(
                (base_ids[candidate_indices], candidate_distances)
            )
            result[start + row_index] = base_ids[candidate_indices[order]]
        print(f"  exact GT: {end}/{len(queries)}", flush=True)
    return result


def write_ground_truth(path: Path, ground_truth: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(ground_truth.tolist())


def load_ground_truth(path: Path) -> list[list[int]]:
    with path.open(newline="") as handle:
        return [[int(value) for value in row] for row in csv.reader(handle)]

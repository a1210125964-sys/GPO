#!/usr/bin/env python3
"""
HNSW cold-cache single-pass read benchmark with a disjoint query set

What it does:
  1. Load the query Parquet strictly disjoint from the 100K base plus its exact ground truth
  2. Verify the full index row count, the dataset's standard opclass, shared_buffers, and Direct I/O
  3. Restart PostgreSQL before each ef_search and optionally drop the OS page cache of relation files
  4. Run EXPLAIN ANALYZE BUFFERS exactly once per query, with no warmup
  5. Compute Recall separately after all performance passes (--skip-recall skips it entirely),
     and write QPS, latency, and buffer metric CSV

Usage:
    python3 bench/bench_read.py \
        --dataset-name Deep100K --variant GPO \
        --commit <resolved-commit-hash> \
        --data-dir data \
        --m 24 --ef-construction 200 --ef-search 40,80,160,200,400 \
        --k 10 --nqueries 1000 --buffersize 39MB \
        --direct-io --drop-cache \
        --output results/GPO_Deep100K_singlepass_39MB.csv

Arguments:
  Data & connection:
    --dataset-name            dataset name                           [required]
    --query-file              disjoint query Parquet                [default: data/queries/<dataset>/query-1000.parquet]
    --ground-truth            exact ground truth CSV                [default: data/queries/<dataset>/ground-truth-k10.csv]
    --dbname                  PostgreSQL database name               [default: iotest]
    --user                    PostgreSQL user                        [default: current system user]
    --host                    PostgreSQL host                        [default: localhost]
    --port                    PostgreSQL port                        [default: 5432]
    --data-dir                PostgreSQL data directory              [required]

  Experiment identity & index:
    --variant                 implementation/layout name             [required]
    --commit                  Git commit                             [required]
    --m                       HNSW M (result metadata)               [default: 24]
    --ef-construction         HNSW build search width (result metadata)  [default: 200]
    --ef-search               query search width list                [default: 40,80,160,200,400]
    --k                       top-k                                  [default: 10]
    --nqueries                use the first N queries from the file  [default: 1000]
    --expected-index-rows     required full base rows in the table   [default: 100000]

  PG config:
    --buffersize              expected shared_buffers and result label  [required]
    --direct-io               require debug_io_direct to include data  [default: False]
    --drop-cache              call posix_fadvise on relation files each round  [default: False]
    --recall-from             reuse Recall from an existing CSV of the same index  [default: measure separately]
    --skip-recall             skip the recall pass; leave recall_at_k empty  [default: False]

  Output:
    --output                  single-pass result CSV                 [required]

Metrics collected:
  Recall@k (empty with --skip-recall), execution/wall QPS, P50/P95/P99,
  average shared hit/read blocks, buffer hit rate, per-pass wall time, and fadvise file count.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import psycopg2

from bench_read_common import (
    DATASETS,
    DatasetSpec,
    load_ground_truth,
    load_parquet_vectors,
)


PG_INSTALL = Path(__file__).resolve().parents[1] / "build" / "src" / "pg_install"


def connect_db(args: argparse.Namespace):
    return psycopg2.connect(
        dbname=args.dbname,
        user=args.user,
        host=args.host,
        port=args.port,
    )


def pg_bin(args: argparse.Namespace, name: str) -> Path:
    return args.pg_install / "bin" / name


def vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def pg_ctl_restart(data_dir: Path, log_path: Path | None = None, pg_install: Path | None = None) -> None:
    pg_ctl = (pg_install or PG_INSTALL) / "bin" / "pg_ctl"
    command = [str(pg_ctl), "restart", "-D", str(data_dir), "-w", "-t", "60"]
    if log_path is None:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command.extend(["-l", str(log_path)])
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def drop_os_page_cache(conn) -> int:
    import ctypes

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    dontneed = 4
    cur = conn.cursor()
    cur.execute("SHOW data_directory")
    data_dir = cur.fetchone()[0]
    cur.close()

    count = 0
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if not filename or not filename[0].isdigit():
                continue
            path = os.path.join(root, filename)
            try:
                fd = os.open(path, os.O_RDONLY)
                libc.posix_fadvise(fd, 0, 0, dontneed)
                os.close(fd)
                count += 1
            except OSError:
                pass
    return count


def has_index_scan(node: dict) -> bool:
    if "Index Scan" in node.get("Node Type", ""):
        return True
    return any(has_index_scan(child) for child in node.get("Plans", []))


def percentile(values: list[float], percentile_value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile_value))


def verify_runtime(
    conn,
    args: argparse.Namespace,
    spec: DatasetSpec,
) -> tuple[int, str, str]:
    cur = conn.cursor()
    cur.execute(f"SELECT count(*) FROM {spec.table}")
    nindex = int(cur.fetchone()[0])
    if nindex != args.expected_index_rows:
        raise RuntimeError(
            f"{spec.table} has {nindex} rows, expected {args.expected_index_rows}"
        )

    cur.execute(
        "SELECT indexdef FROM pg_indexes "
        "WHERE schemaname = current_schema() AND tablename = %s "
        "AND indexdef LIKE '%%USING hnsw%%'",
        (spec.table,),
    )
    definitions = [row[0] for row in cur.fetchall()]
    if not any(spec.operator_class in definition for definition in definitions):
        raise RuntimeError(
            f"{spec.table} has no HNSW index using {spec.operator_class}: {definitions}"
        )

    cur.execute("SHOW debug_io_direct")
    debug_io_direct = cur.fetchone()[0]
    if args.direct_io:
        enabled = {value.strip() for value in debug_io_direct.split(",")}
        if "data" not in enabled:
            raise RuntimeError(
                f"debug_io_direct is {debug_io_direct!r}; expected data Direct I/O"
            )

    cur.execute("SHOW shared_buffers")
    shared_buffers = cur.fetchone()[0]
    cur.execute(
        "SELECT pg_size_bytes(current_setting('shared_buffers')), pg_size_bytes(%s)",
        (args.buffersize,),
    )
    actual_bytes, expected_bytes = cur.fetchone()
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"shared_buffers is {shared_buffers}, expected {args.buffersize}"
        )
    cur.close()
    return nindex, debug_io_direct, shared_buffers


def performance_pass(
    args: argparse.Namespace,
    spec: DatasetSpec,
    vectors: np.ndarray,
    ef_search: int,
) -> dict[str, float | int]:
    pg_ctl_log = None
    if args.pgctl_log_dir:
        pg_ctl_log = args.pgctl_log_dir / f"{spec.name}_ef{ef_search}_postgres.log"
    pg_ctl_restart(args.data_dir, pg_ctl_log, args.pg_install)
    conn = connect_db(args)
    verify_runtime(conn, args, spec)
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET track_io_timing = on")
    cur.execute("SET hnsw.ef_search = %s", (ef_search,))
    cur.close()

    dropped = drop_os_page_cache(conn) if args.drop_cache else 0
    latencies: list[float] = []
    shared_hits = 0
    shared_reads = 0
    wall_start = time.perf_counter()

    for index, vector in enumerate(vectors):
        cur = conn.cursor()
        cur.execute(
            f"""
            EXPLAIN (ANALYZE, BUFFERS, TIMING OFF, FORMAT JSON)
            SELECT id FROM {spec.table}
            ORDER BY embedding {spec.operator} %s::vector
            LIMIT %s
            """,
            (vector_literal(vector), args.k),
        )
        document = cur.fetchone()[0][0]
        cur.close()

        plan = document["Plan"]
        if not has_index_scan(plan):
            raise RuntimeError(f"Query {index} did not use an index scan")
        shared_hits += int(plan.get("Shared Hit Blocks", 0))
        shared_reads += int(plan.get("Shared Read Blocks", 0))
        latencies.append(float(document["Execution Time"]))

        if (index + 1) % 25 == 0 or index + 1 == len(vectors):
            print(
                f"  performance ef={ef_search}: {index + 1}/{len(vectors)}",
                flush=True,
            )

    wall_seconds = time.perf_counter() - wall_start
    conn.close()
    total_blocks = shared_hits + shared_reads
    return {
        "execution_qps": 1000.0 * len(latencies) / sum(latencies),
        "wall_qps": len(latencies) / wall_seconds,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
        "avg_buffers_hit": shared_hits / len(latencies),
        "avg_buffers_read": shared_reads / len(latencies),
        "buffer_hit_rate": shared_hits / total_blocks if total_blocks else 1.0,
        "wall_seconds": wall_seconds,
        "cache_files_advised": dropped,
    }


def recall_pass(
    args: argparse.Namespace,
    spec: DatasetSpec,
    vectors: np.ndarray,
    ground_truth: list[list[int]],
    ef_search: int,
) -> float:
    conn = connect_db(args)
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET hnsw.ef_search = %s", (ef_search,))
    recalls = []
    for index, vector in enumerate(vectors):
        cur.execute(
            f"""
            SELECT id FROM {spec.table}
            ORDER BY embedding {spec.operator} %s::vector
            LIMIT %s
            """,
            (vector_literal(vector), args.k),
        )
        result = {row[0] for row in cur.fetchall()}
        expected = set(ground_truth[index][: args.k])
        recalls.append(len(result & expected) / args.k)
    cur.close()
    conn.close()
    return float(np.mean(recalls))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True, choices=sorted(DATASETS))
    parser.add_argument("--query-file", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--dbname", default="iotest")
    parser.add_argument("--user", default=os.environ.get("USER"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=5432, type=int)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--pg-install",
        type=Path,
        default=PG_INSTALL,
        help="PostgreSQL install tree for pg_ctl (page-size experiment clusters)",
    )
    parser.add_argument(
        "--pgctl-log-dir",
        type=Path,
        help="Directory for pg_ctl restart logs; each performance pass uses -l",
    )
    parser.add_argument("--m", default=24, type=int)
    parser.add_argument("--ef-construction", default=200, type=int)
    parser.add_argument("--ef-search", default="40,80,160,200,400")
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--nqueries", default=1000, type=int)
    parser.add_argument("--expected-index-rows", default=100_000, type=int)
    parser.add_argument("--buffersize", required=True)
    parser.add_argument("--direct-io", action="store_true")
    parser.add_argument("--drop-cache", action="store_true")
    parser.add_argument(
        "--recall-from",
        type=Path,
        help=(
            "Reuse Recall values from a completed CSV for the same index and "
            "ef_search values. Recall is independent of shared_buffers."
        ),
    )
    parser.add_argument(
        "--skip-recall",
        action="store_true",
        help=(
            "Skip the recall pass entirely; recall_at_k is written as an "
            "empty field (main read-benchmark protocol, 2026-09-15)."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.skip_recall and args.recall_from:
        parser.error("--skip-recall and --recall-from are mutually exclusive")
    return args


def main() -> int:
    args = parse_args()
    spec = DATASETS[args.dataset_name]
    query_file = args.query_file or spec.query_file
    ground_truth_file = args.ground_truth or spec.ground_truth_file

    vectors = load_parquet_vectors(query_file, "embedding", spec.dimension)
    if args.nqueries <= 0 or args.nqueries > len(vectors):
        raise ValueError(
            f"nqueries must be in 1..{len(vectors)}, got {args.nqueries}"
        )
    vectors = vectors[: args.nqueries]
    ground_truth = load_ground_truth(ground_truth_file)
    if len(ground_truth) < len(vectors):
        raise RuntimeError(
            f"GT has {len(ground_truth)} rows, queries have {len(vectors)}"
        )
    ground_truth = ground_truth[: len(vectors)]

    conn = connect_db(args)
    nindex, debug_io_direct, shared_buffers = verify_runtime(conn, args, spec)
    conn.close()

    ef_values = [int(value) for value in args.ef_search.split(",")]
    results: list[dict[str, float | int]] = []
    for ef_search in ef_values:
        print(
            f"Starting cold single-pass: {spec.name}, ef={ef_search}",
            flush=True,
        )
        metrics = performance_pass(args, spec, vectors, ef_search)
        metrics["ef_search"] = ef_search
        results.append(metrics)

    # Recall is deliberately measured after all performance sweeps. For a
    # shared_buffers sweep over one immutable index, it can be copied from a
    # prior run because buffer capacity cannot change returned neighbors.
    # --skip-recall (main read-benchmark protocol) leaves recall_at_k empty.
    if args.skip_recall:
        for row in results:
            row["recall_at_k"] = ""
    elif args.recall_from:
        with args.recall_from.open(newline="") as handle:
            prior_rows = list(csv.DictReader(handle))
        prior_recall: dict[int, float] = {}
        for prior in prior_rows:
            if (
                prior["dataset"] != spec.name
                or prior["variant"] != args.variant
                or prior["commit"] != args.commit
                or int(prior["nqueries"]) != len(vectors)
                or int(prior["k"]) != args.k
            ):
                raise RuntimeError(
                    f"Incompatible recall source row in {args.recall_from}: {prior}"
                )
            prior_recall[int(prior["ef_search"])] = float(prior["recall_at_k"])
        missing = sorted(set(ef_values) - set(prior_recall))
        if missing:
            raise RuntimeError(
                f"Recall source {args.recall_from} is missing ef_search={missing}"
            )
        for row in results:
            row["recall_at_k"] = prior_recall[int(row["ef_search"])]
    else:
        for row in results:
            row["recall_at_k"] = recall_pass(
                args, spec, vectors, ground_truth, int(row["ef_search"])
            )

    fields = [
        "dataset",
        "variant",
        "commit",
        "metric",
        "nindex",
        "nqueries",
        "m",
        "ef_construction",
        "ef_search",
        "k",
        "buffersize",
        "shared_buffers_actual",
        "direct_io",
        "debug_io_direct_actual",
        "recall_at_k",
        "execution_qps",
        "wall_qps",
        "wall_seconds",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "avg_buffers_hit",
        "avg_buffers_read",
        "buffer_hit_rate",
        "cache_files_advised",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metrics in results:
            writer.writerow(
                {
                    "dataset": spec.name,
                    "variant": args.variant,
                    "commit": args.commit,
                    "metric": spec.metric,
                    "nindex": nindex,
                    "nqueries": len(vectors),
                    "m": args.m,
                    "ef_construction": args.ef_construction,
                    "ef_search": metrics["ef_search"],
                    "k": args.k,
                    "buffersize": args.buffersize,
                    "shared_buffers_actual": shared_buffers,
                    "direct_io": "on" if args.direct_io else "off",
                    "debug_io_direct_actual": debug_io_direct,
                    "recall_at_k": (
                        f"{metrics['recall_at_k']:.6f}"
                        if metrics["recall_at_k"] != ""
                        else ""
                    ),
                    "execution_qps": f"{metrics['execution_qps']:.6f}",
                    "wall_qps": f"{metrics['wall_qps']:.6f}",
                    "wall_seconds": f"{metrics['wall_seconds']:.6f}",
                    "latency_p50_ms": f"{metrics['latency_p50_ms']:.6f}",
                    "latency_p95_ms": f"{metrics['latency_p95_ms']:.6f}",
                    "latency_p99_ms": f"{metrics['latency_p99_ms']:.6f}",
                    "avg_buffers_hit": f"{metrics['avg_buffers_hit']:.6f}",
                    "avg_buffers_read": f"{metrics['avg_buffers_read']:.6f}",
                    "buffer_hit_rate": f"{metrics['buffer_hit_rate']:.6f}",
                    "cache_files_advised": metrics["cache_files_advised"],
                }
            )
    print(f"Saved: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

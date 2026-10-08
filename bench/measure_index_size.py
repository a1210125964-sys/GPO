#!/usr/bin/env python3
"""
HNSW full-index build and index size measurement

What it does:
  1. Load the full 100K base vectors per dataset in a fixed order and import them into PostgreSQL
  2. Build HNSW with the dataset's standard distance and the given m and ef_construction
  3. Sample peak RSS/PSS of the CREATE INDEX backend and its parallel workers
  4. Measure the HNSW index with pg_relation_size only, excluding table, TOAST, and other indexes
  5. Derive suggested shared_buffers values of 10%/20%/30%/40%/50% of the index size
  6. Optionally keep built tables for reuse by read experiments and write build info to CSV

Usage:
    python3 bench/measure_index_size.py \
        --datasets Deep100K \
        --m 24 --ef-construction 200 --keep-tables \
        --require-direct-io --expected-shared-buffers 128MB \
        --variant baseline --commit <resolved-commit-hash> \
        --output results/baseline_index_sizes.csv

Arguments:
  Datasets & connection:
    --datasets                 datasets to build                     [default: Deep100K]
    --dbname                  PostgreSQL database name               [default: iotest]
    --user                    PostgreSQL user                        [default: current system user]
    --host                    PostgreSQL host                        [default: localhost]
    --port                    PostgreSQL port                        [default: 5432]

  Index:
    --m                       HNSW M                                 [default: 24]
    --ef-construction         HNSW build search width                [default: 200]
    --enable-aftp             enable hnsw.aftp_enable in the CREATE INDEX session  [default: False]
    --aftp-window             hnsw.aftp_window for the CREATE INDEX session       [default: not set]
    --aftp-sample-queries     hnsw.aftp_sample_queries for the CREATE INDEX session  [default: not set]
    --keep-tables             keep tables and indexes after measurement  [default: False]
    --reuse-existing          validate and reuse an existing full table and index  [default: False]
    --memory-sample-ms        build memory sampling interval in milliseconds  [default: 50]

  PG config checks:
    --require-direct-io       require debug_io_direct to include data  [default: False]
    --expected-shared-buffers require shared_buffers to equal the given value  [default: no check]

  Output metadata:
    --variant                 implementation/layout name             [default: unknown]
    --commit                  Git commit                             [default: unknown]
    --output                  index size and buffer suggestion CSV   [default: no CSV written]
"""

from __future__ import annotations

import argparse
import csv
import os
import threading
import time
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from bench_read_common import DATASETS, DatasetSpec, load_parquet_vectors


BATCH_SIZE = 2_000
BUFFER_PERCENTAGES = (10, 20, 30, 40, 50)
DEFAULT_MEMORY_SAMPLE_MS = 50


def connect(args: argparse.Namespace):
    return psycopg2.connect(
        dbname=args.dbname,
        user=args.user,
        host=args.host,
        port=args.port,
    )


def vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def read_proc_memory(pid: int) -> tuple[int | None, int | None]:
    """Return Linux RSS and PSS bytes for one process."""
    rss_bytes = None
    pss_bytes = None

    try:
        with Path(f"/proc/{pid}/status").open() as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss_bytes = int(line.split()[1]) * 1024
                    break
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None, None

    try:
        with Path(f"/proc/{pid}/smaps_rollup").open() as handle:
            for line in handle:
                if line.startswith("Pss:"):
                    pss_bytes = int(line.split()[1]) * 1024
                    break
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass

    return rss_bytes, pss_bytes


class BuildMemoryMonitor:
    """Sample memory for a CREATE INDEX leader and its parallel workers."""

    def __init__(
        self,
        args: argparse.Namespace,
        leader_pid: int,
        sample_interval_ms: int,
    ) -> None:
        self.args = args
        self.leader_pid = leader_pid
        self.sample_interval_seconds = sample_interval_ms / 1000.0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.samples = 0
        self.peak_processes = 0
        self.worker_discovery_available = True
        self.error: str | None = None

        baseline_rss, baseline_pss = read_proc_memory(leader_pid)
        self.baseline_rss_bytes = baseline_rss
        self.baseline_pss_bytes = baseline_pss
        self.peak_rss_bytes = baseline_rss
        self.peak_pss_bytes = baseline_pss

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run,
            name=f"build-memory-monitor-{self.leader_pid}",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> dict[str, int | str]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()

        if self.error:
            print(f"  memory monitor warning: {self.error}", flush=True)

        return {
            "build_backend_pid": self.leader_pid,
            "build_memory_samples": self.samples,
            "build_peak_processes": self.peak_processes,
            "build_rss_before_bytes": self._value(self.baseline_rss_bytes),
            "build_peak_rss_bytes": self._value(self.peak_rss_bytes),
            "build_peak_rss_delta_bytes": self._delta(
                self.peak_rss_bytes, self.baseline_rss_bytes
            ),
            "build_pss_before_bytes": self._value(self.baseline_pss_bytes),
            "build_peak_pss_bytes": self._value(self.peak_pss_bytes),
            "build_peak_pss_delta_bytes": self._delta(
                self.peak_pss_bytes, self.baseline_pss_bytes
            ),
            "build_peak_pss_mib": self._mib(self.peak_pss_bytes),
            "build_peak_pss_delta_mib": self._mib_delta(
                self.peak_pss_bytes, self.baseline_pss_bytes
            ),
        }

    @staticmethod
    def _value(value: int | None) -> int | str:
        return "" if value is None else value

    @staticmethod
    def _delta(peak: int | None, baseline: int | None) -> int | str:
        if peak is None or baseline is None:
            return ""
        return max(0, peak - baseline)

    @staticmethod
    def _mib(value: int | None) -> str:
        return "" if value is None else f"{value / 1024 / 1024:.6f}"

    @staticmethod
    def _mib_delta(peak: int | None, baseline: int | None) -> str:
        if peak is None or baseline is None:
            return ""
        return f"{max(0, peak - baseline) / 1024 / 1024:.6f}"

    def _backend_pids(self, conn) -> list[int]:
        if not self.worker_discovery_available:
            return [self.leader_pid]

        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT pid FROM pg_stat_activity "
                "WHERE pid = %s OR leader_pid = %s",
                (self.leader_pid, self.leader_pid),
            )
            pids = [int(row[0]) for row in cur.fetchall()]
            cur.close()
            return pids or [self.leader_pid]
        except psycopg2.Error:
            conn.rollback()
            self.worker_discovery_available = False
            return [self.leader_pid]

    def _sample(self, conn) -> None:
        pids = self._backend_pids(conn)
        rss_values: list[int] = []
        pss_values: list[int] = []

        for pid in pids:
            rss_bytes, pss_bytes = read_proc_memory(pid)
            if rss_bytes is not None:
                rss_values.append(rss_bytes)
            if pss_bytes is not None:
                pss_values.append(pss_bytes)

        if rss_values:
            total_rss = sum(rss_values)
            if self.peak_rss_bytes is None or total_rss > self.peak_rss_bytes:
                self.peak_rss_bytes = total_rss

        # Only aggregate PSS when every discovered process was readable.
        if pss_values and len(pss_values) == len(pids):
            total_pss = sum(pss_values)
            if self.peak_pss_bytes is None or total_pss > self.peak_pss_bytes:
                self.peak_pss_bytes = total_pss

        self.peak_processes = max(self.peak_processes, len(rss_values))
        self.samples += 1

    def _run(self) -> None:
        conn = None
        try:
            conn = connect(self.args)
            conn.autocommit = True
            while not self.stop_event.is_set():
                self._sample(conn)
                self.stop_event.wait(self.sample_interval_seconds)
            self._sample(conn)
        except Exception as exc:  # Monitoring must not abort a successful build.
            self.error = str(exc)
        finally:
            if conn is not None:
                conn.close()


def empty_build_memory_metrics() -> dict[str, str]:
    return {
        "build_backend_pid": "",
        "build_memory_samples": "",
        "build_peak_processes": "",
        "build_rss_before_bytes": "",
        "build_peak_rss_bytes": "",
        "build_peak_rss_delta_bytes": "",
        "build_pss_before_bytes": "",
        "build_peak_pss_bytes": "",
        "build_peak_pss_delta_bytes": "",
        "build_peak_pss_mib": "",
        "build_peak_pss_delta_mib": "",
    }


def verify_server_settings(conn, args: argparse.Namespace) -> dict[str, str]:
    cur = conn.cursor()
    cur.execute("SHOW shared_buffers")
    shared_buffers = cur.fetchone()[0]
    cur.execute("SHOW debug_io_direct")
    debug_io_direct = cur.fetchone()[0]
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    row = cur.fetchone()
    vector_version = row[0] if row else "not-installed"

    if args.expected_shared_buffers:
        cur.execute(
            "SELECT pg_size_bytes(current_setting('shared_buffers')), "
            "pg_size_bytes(%s)",
            (args.expected_shared_buffers,),
        )
        actual_bytes, expected_bytes = cur.fetchone()
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"shared_buffers is {shared_buffers}, expected "
                f"{args.expected_shared_buffers}"
            )
    if args.require_direct_io:
        enabled = {value.strip() for value in debug_io_direct.split(",")}
        if "data" not in enabled:
            raise RuntimeError(
                f"debug_io_direct is {debug_io_direct!r}; data Direct I/O is required"
            )
    cur.close()
    return {
        "shared_buffers": shared_buffers,
        "debug_io_direct": debug_io_direct,
        "vector_version": vector_version,
    }


def import_vectors(conn, table: str, vectors: np.ndarray) -> None:
    cur = conn.cursor()
    inserted = 0
    started = time.perf_counter()
    for start in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[start : start + BATCH_SIZE]
        execute_values(
            cur,
            f"INSERT INTO {table} (embedding) VALUES %s",
            [(vector_literal(vector),) for vector in batch],
            template="(%s::vector)",
            page_size=BATCH_SIZE,
        )
        conn.commit()
        inserted += len(batch)
        if inserted % 20_000 == 0 or inserted == len(vectors):
            print(
                f"  import: {inserted}/{len(vectors)} "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
    cur.close()


def build_and_measure(
    conn,
    spec: DatasetSpec,
    args: argparse.Namespace,
) -> dict[str, object]:
    print(
        f"[{spec.name}] full base / {spec.metric} / "
        f"m={args.m} / ef_construction={args.ef_construction}",
        flush=True,
    )
    vectors = load_parquet_vectors(
        spec.base_dir, spec.base_column, spec.dimension
    )
    if len(vectors) != 100_000:
        raise ValueError(
            f"{spec.name}: expected 100,000 base rows, got {len(vectors)}"
        )
    zero_vectors = int(np.count_nonzero(np.linalg.norm(vectors, axis=1) == 0))
    indexable_rows = len(vectors) - zero_vectors if spec.metric == "cosine" else len(vectors)
    index_name = f"{spec.table}_hnsw_idx"

    cur = conn.cursor()
    cur.execute(
        "SELECT to_regclass(%s), to_regclass(%s)",
        (spec.table, index_name),
    )
    table_relation, index_relation = cur.fetchone()
    reuse_existing = bool(
        args.reuse_existing and table_relation and index_relation
    )
    if reuse_existing:
        cur.execute(f"SELECT count(*) FROM {spec.table}")
        existing_rows = int(cur.fetchone()[0])
        cur.execute("SELECT pg_get_indexdef(%s::regclass)", (index_name,))
        index_definition = cur.fetchone()[0]
        if existing_rows != len(vectors):
            raise RuntimeError(
                f"Cannot reuse {spec.table}: {existing_rows} rows, expected {len(vectors)}"
            )
        if spec.operator_class not in index_definition:
            raise RuntimeError(
                f"Cannot reuse {index_name}: expected {spec.operator_class}"
            )
        build_seconds = None
        memory_metrics: dict[str, int | str] = empty_build_memory_metrics()
        print(f"  reusing validated table and index: {index_name}", flush=True)
    else:
        cur.execute(f"DROP TABLE IF EXISTS {spec.table} CASCADE")
        cur.execute(
            f"CREATE TABLE {spec.table} ("
            f"id serial PRIMARY KEY, embedding vector({spec.dimension}))"
        )
        conn.commit()
        cur.close()

        import_vectors(conn, spec.table, vectors)

        cur = conn.cursor()
        if args.enable_aftp:
            conn.notices.clear()
            cur.execute("SET client_min_messages = info")
            cur.execute("SET hnsw.aftp_enable = on")
            for guc_name, value in (
                ("hnsw.aftp_window", args.aftp_window),
                ("hnsw.aftp_sample_queries", args.aftp_sample_queries),
            ):
                if value is not None:
                    cur.execute(f"SET {guc_name} = %s", (value,))
            cur.execute("SHOW hnsw.aftp_enable")
            if cur.fetchone()[0] != "on":
                raise RuntimeError("Failed to enable hnsw.aftp_enable")

        cur.execute("SELECT pg_backend_pid()")
        backend_pid = int(cur.fetchone()[0])
        memory_monitor = BuildMemoryMonitor(
            args, backend_pid, args.memory_sample_ms
        )
        memory_monitor.start()
        started = time.perf_counter()
        try:
            cur.execute(
                f"CREATE INDEX {index_name} ON {spec.table} "
                f"USING hnsw (embedding {spec.operator_class}) "
                f"WITH (m = {args.m}, ef_construction = {args.ef_construction})"
            )
            conn.commit()
        finally:
            build_seconds = time.perf_counter() - started
            memory_metrics = memory_monitor.stop()
            if args.enable_aftp:
                # psycopg2 queues server NOTICE/INFO responses here; AFTP
                # phase timings arrive only with client_min_messages=info.
                for notice in conn.notices:
                    print(f"PGNOTICE: {notice.rstrip()}", flush=True)
                conn.notices.clear()

    cur.execute("SELECT pg_relation_size(%s::regclass)", (index_name,))
    index_bytes = int(cur.fetchone()[0])
    cur.execute("SELECT pg_size_pretty(%s::bigint)", (index_bytes,))
    index_pretty = cur.fetchone()[0]
    cur.execute(f"SELECT count(*) FROM {spec.table}")
    table_rows = int(cur.fetchone()[0])
    cur.close()
    if table_rows != len(vectors):
        raise RuntimeError(
            f"{spec.table}: table has {table_rows} rows, expected {len(vectors)}"
        )

    index_mib = index_bytes / 1024 / 1024
    buffers = {
        percentage: max(1, int(index_mib * percentage / 100 + 0.5))
        for percentage in BUFFER_PERCENTAGES
    }
    print(
        f"  HNSW index: {index_pretty} ({index_bytes} bytes), "
        f"build={'reused' if build_seconds is None else f'{build_seconds:.3f}s'}, "
        f"zero_vectors={zero_vectors}",
        flush=True,
    )
    if memory_metrics["build_peak_pss_mib"] != "":
        print(
            "  build memory: "
            f"peak PSS={memory_metrics['build_peak_pss_mib']} MiB, "
            f"delta={memory_metrics['build_peak_pss_delta_mib']} MiB, "
            f"peak processes={memory_metrics['build_peak_processes']}, "
            f"samples={memory_metrics['build_memory_samples']}",
            flush=True,
        )
    print(
        "  buffers: "
        + ", ".join(f"{percentage}%={value}MB" for percentage, value in buffers.items()),
        flush=True,
    )

    if not args.keep_tables:
        cur = conn.cursor()
        cur.execute(f"DROP TABLE {spec.table} CASCADE")
        conn.commit()
        cur.close()

    return {
        "dataset": spec.name,
        "metric": spec.metric,
        "dimension": spec.dimension,
        "table_name": spec.table,
        "base_rows": len(vectors),
        "indexable_rows": indexable_rows,
        "zero_vectors": zero_vectors,
        "m": args.m,
        "ef_construction": args.ef_construction,
        "index_bytes": index_bytes,
        "index_mib": f"{index_mib:.6f}",
        "build_seconds": "" if build_seconds is None else f"{build_seconds:.6f}",
        **memory_metrics,
        "reused_existing": reuse_existing,
        **{f"buffer_{percentage}pct_mb": buffers[percentage] for percentage in BUFFER_PERCENTAGES},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS), default=["Deep100K"]
    )
    parser.add_argument("--dbname", default="iotest")
    parser.add_argument("--user", default=os.environ.get("USER"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--m", type=int, default=24)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--keep-tables", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--enable-aftp", action="store_true")
    parser.add_argument(
        "--aftp-window",
        type=int,
        help="SET hnsw.aftp_window in the CREATE INDEX session",
    )
    parser.add_argument(
        "--aftp-sample-queries",
        type=int,
        help="SET hnsw.aftp_sample_queries in the CREATE INDEX session",
    )
    parser.add_argument(
        "--memory-sample-ms",
        default=DEFAULT_MEMORY_SAMPLE_MS,
        type=int,
        help="CREATE INDEX backend/worker memory sampling interval in milliseconds",
    )
    parser.add_argument("--require-direct-io", action="store_true")
    parser.add_argument("--expected-shared-buffers")
    parser.add_argument("--variant", default="unknown")
    parser.add_argument("--commit", default="unknown")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.memory_sample_ms <= 0:
        raise ValueError("memory-sample-ms must be positive")
    conn = connect(args)
    try:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        cur.close()
        settings = verify_server_settings(conn, args)
        print(f"server settings: {settings}", flush=True)

        rows = [build_and_measure(conn, DATASETS[name], args) for name in args.datasets]
    finally:
        conn.close()

    for row in rows:
        row["variant"] = args.variant
        row["commit"] = args.commit
        row.update(settings)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

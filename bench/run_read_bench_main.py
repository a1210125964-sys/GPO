#!/usr/bin/env python3
"""
Main experiment: cold-cache single-pass read performance, 5 variants × the Deep100K dataset (single 50% buffer tier)

What it does:
  1. Installs the five variants in turn: baseline/TurboCharging/SEN/ATL/GPO (no seedfix applied)
  2. Per variant, under 128MB + data Direct I/O + serial build (4GB maintenance),
     rebuilds the Deep100K index for this page size via bench/measure_index_size.py (AFTP
     variants SET aftp_enable=on, aftp_window=32, aftp_sample_queries=400 per
     the E1 settings, index-build session only)
  3. The buffer tier is uniformly 50% of the baseline (pg_orig) index size
     (buffer_50pct_mb); the same absolute MB is shared by all five variants
     for a given dataset
  4. Per combo: ALTER SYSTEM to {N}MB, restart, then invoke bench/bench_read.py
     (ef_search=200, 1000 queries, --direct-io --drop-cache --skip-recall)
  5. TurboCharging cannot build at 4KB pages; only relevant if --page-size 4096 is used
  6. --resume skips completed combos via CSV integrity checks; at the end the
     repository branch is restored, pgvector reinstalled, and the cluster
     config reverted (RESET debug_io_direct, 128MB)

Usage:
    python3 bench/run_read_bench_main.py \
        --result-dir "results/main" \
        --page-size 8192 --resume

    The repository ships only the Deep100K dataset; pass --datasets to run
    others after staging them under data/ (see bench/README.md).

Arguments:
  Experiment matrix:
    --page-size      page size in bytes                    [default: 8192]
    --variants       variants to run                       [default: all five]
    --datasets       datasets to run                       [default: Deep100K]
    --resume         skip combos that passed integrity checks [default: False]

  Paths:
    --result-dir     experiment result directory (main experiment/) [required]
    --repo-dir       GPO build worktree (build/src)        [default: fixed project path]
    --keep-source    do not restore the pre-experiment branch at the end (for debugging) [default: False]

Result layout (follows the result-directory organization convention):
    <result-dir>/pagesize_<N>KB/
        index_sizes/<variant>_index_sizes.csv
        <variant>/<commit4>_<dataset>/read_bench_BufferSize<N>MB.csv
        read_bench_main_summary.csv
        logs/

Collected metrics (bench_read.py fields):
  execution/wall QPS, P50/P95/P99, avg shared hit/read blocks,
  buffer hit rate, wall time, fadvise file count; recall_at_k left empty (--skip-recall).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
MEASURE_SCRIPT = PROJECT_DIR / "bench" / "measure_index_size.py"
BENCH_SCRIPT = PROJECT_DIR / "bench" / "bench_read.py"

M = 24
EF_CONSTRUCTION = 200
EF_SEARCH = 200
NQUERIES = 1000
BUILD_SHARED_BUFFERS = "128MB"
AFTP_WINDOW = 32
AFTP_SAMPLES = 400
DBNAME = "iotest"

# (label, branch, aftp) — commits are resolved from the branch names at
# runtime so the driver never hardcodes hashes.
VERSIONS: list[tuple[str, str, bool]] = [
    ("baseline", "pg_orig", False),
    ("TurboCharging", "Turbocharging", False),
    ("SEN", "SEN", False),
    ("ATL", "ATL", True),
    ("GPO", "GPO", True),
]


def resolve_commit(repo_dir: Path, branch: str) -> str:
    """Resolve a branch name to its full commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", branch],
        cwd=repo_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot resolve branch '{branch}' in {repo_dir}: {result.stderr.strip()}")
    return result.stdout.strip()
AFTP_LABELS = {"ATL", "GPO"}
DATASET_ORDER = ["Deep100K"]  # single-dataset artifact; extend after staging data/
PAGE_SIZE_ORDER = [8192, 4096, 16384]
PAGE_SIZE_DIR = {4096: "pagesize_4KB", 8192: "pagesize_8KB",
                 16384: "pagesize_16KB"}

# TC's 8004B partitioned page layout necessarily overflows on 4KB pages
# (pd_lower>4096 means ERRCODE_DATA_CORRUPTED); the Page Utilization
# experiment already reproduced this 5/5.
EXCLUDED_COMBOS = {(4096, "TurboCharging")}

SUMMARY_NAME = "read_bench_main_summary.csv"


@dataclass(frozen=True)
class Cluster:
    pg_install: Path
    pgdata: Path
    port: int


def cluster_for(repo_dir: Path, page_size: int) -> Cluster:
    if page_size == 8192:
        return Cluster(repo_dir / "pg_install", repo_dir / "pgdata", 5432)
    if page_size == 4096:
        return Cluster(repo_dir / "pg_install_4k", repo_dir / "pgdata_4k", 5434)
    if page_size == 16384:
        return Cluster(repo_dir / "pg_install_16k", repo_dir / "pgdata_16k", 5433)
    raise ValueError(f"unsupported page size: {page_size}")


# ---------------------------------------------------------------------------
# Execution helpers (machinery reused from util/23_run_page_utilization_pagesize.py)
# ---------------------------------------------------------------------------

def command_environment(cluster: Cluster) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PATH"] = (
        str(cluster.pg_install / "bin") + os.pathsep + environment.get("PATH", "")
    )
    return environment


def run_quiet(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def run_stream(
    command: list[str], log_path: Path, *, cwd: Path, environment: dict[str, str]
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def git_status_lines(repo_dir: Path) -> list[str]:
    output = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_dir,
        capture_output=True, text=True, check=True,
    ).stdout
    return [line.strip() for line in output.splitlines() if line.strip()]


def git_head_ref(repo_dir: Path) -> str:
    """Return the ref to restore: the branch name, or the commit hash when detached."""
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if branch != "HEAD":
        return branch
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def pg_ctl(cluster: Cluster) -> Path:
    return cluster.pg_install / "bin" / "pg_ctl"


def pg_is_running(cluster: Cluster) -> bool:
    return subprocess.run(
        [str(pg_ctl(cluster)), "status", "-D", str(cluster.pgdata)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def start_server(cluster: Cluster, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(pg_ctl(cluster)), "start", "-D", str(cluster.pgdata),
         "-l", str(log_file), "-w", "-t", "60"],
        check=True,
    )


def stop_server(cluster: Cluster) -> None:
    """Best-effort stop: do not raise if the cluster is already down (e.g. an
    external shutdown), so the finally path does not mask the real failure."""
    result = subprocess.run(
        [str(pg_ctl(cluster)), "stop", "-D", str(cluster.pgdata),
         "-m", "fast", "-w", "-t", "60"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 and pg_is_running(cluster):
        raise RuntimeError(f"pg_ctl stop failed: {result.stderr.strip()}")


def psql_statement(cluster: Cluster, statement: str) -> None:
    psql = cluster.pg_install / "bin" / "psql"
    run_quiet(
        [str(psql), "-h", "localhost", "-p", str(cluster.port),
         "-d", "postgres", "-v", "ON_ERROR_STOP=1", "-c", statement],
        cwd=cluster.pgdata,
    )


def configure_server(cluster: Cluster, work_dir: Path, page_size: int) -> None:
    """Uniform build-time config for the read experiment: shared_buffers=128MB,
    maintenance_work_mem=4GB, max_parallel_maintenance_workers=0 (serial
    build), debug_io_direct='data'. First verify the cluster block_size
    matches --page-size; ALTER SYSTEM takes effect on the next restart, which
    the subsequent start/stop cycle applies naturally."""
    psql = cluster.pg_install / "bin" / "psql"
    temp_log = work_dir / "logs" / "config_server.log"
    was_running = pg_is_running(cluster)
    if not was_running:
        start_server(cluster, temp_log)
    try:
        probe = subprocess.run(
            [str(psql), "-h", "localhost", "-p", str(cluster.port),
             "-d", "postgres", "-A", "-t", "-c",
             "SELECT current_setting('block_size')"],
            capture_output=True, text=True, check=True,
        )
        actual = int(probe.stdout.strip())
        if actual != page_size:
            raise RuntimeError(
                f"cluster block_size={actual} does not match --page-size={page_size}")
        for statement in (
            "ALTER SYSTEM SET shared_buffers = '128MB'",
            "ALTER SYSTEM SET maintenance_work_mem = '4GB'",
            "ALTER SYSTEM SET max_parallel_maintenance_workers = '0'",
            "ALTER SYSTEM SET debug_io_direct = 'data'",
        ):
            psql_statement(cluster, statement)
    finally:
        if not was_running:
            stop_server(cluster)


def build_install(cluster: Cluster, log_dir: Path, label: str,
                  environment: dict[str, str], repo_dir: Path) -> None:
    for step in ("clean", f"-j{os.cpu_count() or 1}", "install"):
        run_stream(
            ["make", step],
            log_dir / f"{label}_make_{step}.log",
            cwd=repo_dir / "pgvector", environment=environment,
        )


def install_version(
    repo_dir: Path, cluster: Cluster, log_dir: Path, label: str, commit: str,
    environment: dict[str, str],
) -> None:
    dirty = git_status_lines(repo_dir)
    if dirty:
        raise RuntimeError(f"repository working tree is dirty, refusing to continue:\n{dirty}")
    run_quiet(["git", "reset", "--hard", "HEAD"], cwd=repo_dir)
    run_quiet(["git", "checkout", "--detach", commit], cwd=repo_dir)
    build_install(cluster, log_dir, label, environment, repo_dir)


def restore_repo(
    repo_dir: Path, cluster: Cluster, log_dir: Path, ref: str,
    environment: dict[str, str], log_prefix: str,
) -> None:
    print(f"=== restoring {ref} and reinstalling pgvector into the target cluster ===", flush=True)
    if pg_is_running(cluster):
        stop_server(cluster)
    run_quiet(["git", "reset", "--hard", "HEAD"], cwd=repo_dir)
    run_quiet(["git", "checkout", ref], cwd=repo_dir)
    build_install(cluster, log_dir, f"{log_prefix}_restore", environment, repo_dir)


def log_contains(path: Path, needle: str, chunk_size: int = 8 * 1024 * 1024) -> bool:
    encoded = needle.encode()
    tail = b""
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                return False
            if encoded in (tail + chunk):
                return True
            tail = (tail + chunk)[-len(encoded):]


# ---------------------------------------------------------------------------
# Index building (measure_index_size.py) and the query benchmark (bench_read.py)
# ---------------------------------------------------------------------------

def index_csv_path(work_dir: Path, label: str) -> Path:
    return work_dir / "index_sizes" / f"{label}_index_sizes.csv"


def bench_csv_path(work_dir: Path, label: str, commit: str,
                   dataset: str, buffer_mb: int) -> Path:
    return (work_dir / label / f"{commit[:4]}_{dataset}"
            / f"read_bench_BufferSize{buffer_mb}MB.csv")


def index_csv_complete(path: Path, datasets: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False
    return sorted(row["dataset"] for row in rows) == sorted(datasets)


def valid_bench_csv(
    path: Path, dataset: str, label: str, commit: str,
    buffer_mb: int, ef_search: int, nqueries: int,
) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return (
            len(rows) == 1
            and rows[0]["dataset"] == dataset
            and rows[0]["variant"] == label
            and rows[0]["commit"] == commit
            and int(rows[0]["ef_search"]) == ef_search
            and int(rows[0]["nqueries"]) == nqueries
            and rows[0]["buffersize"] == f"{buffer_mb}MB"
            and rows[0]["direct_io"] == "on"
            and rows[0]["debug_io_direct_actual"] == "data"
            and rows[0]["recall_at_k"] == ""
            and float(rows[0]["execution_qps"]) > 0.0
        )
    except (KeyError, ValueError, csv.Error, OSError):
        return False


def read_buffer_map(path: Path, datasets: list[str]) -> dict[str, int]:
    """Read the buffer_50pct_mb column of the baseline index-size CSV as the uniform tier."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {row["dataset"]: int(row["buffer_50pct_mb"]) for row in rows}
    missing = [name for name in datasets if name not in mapping]
    if missing:
        raise ValueError(f"{path} is missing dataset rows: {missing}")
    return mapping


def build_indexes(
    args: argparse.Namespace, cluster: Cluster, label: str, commit: str,
    aftp: bool, work_dir: Path, log_dir: Path, server_log: Path,
    environment: dict[str, str],
) -> None:
    output = index_csv_path(work_dir, label)
    command = [
        sys.executable, str(MEASURE_SCRIPT),
        "--datasets", *args.datasets,
        "--host", "localhost", "--port", str(cluster.port),
        "--dbname", DBNAME,
        "--m", str(M), "--ef-construction", str(EF_CONSTRUCTION),
        "--keep-tables", "--require-direct-io",
        "--expected-shared-buffers", BUILD_SHARED_BUFFERS,
        "--variant", label, "--commit", commit,
        "--output", str(output),
    ]
    if aftp:
        command += [
            "--enable-aftp",
            "--aftp-window", str(AFTP_WINDOW),
            "--aftp-sample-queries", str(AFTP_SAMPLES),
        ]
    run_stream(
        command, log_dir / f"{label}_index_build.log",
        cwd=PROJECT_DIR, environment=environment,
    )
    if not index_csv_complete(output, args.datasets):
        raise RuntimeError(f"index build output incomplete: {output}")
    if aftp and log_contains(server_log, "falling back to default layout"):
        raise RuntimeError(
            f"{label}: AFTP build fell back to the native layout; results cannot serve as a comparison; aborting.")


def set_shared_buffers(cluster: Cluster, buffer_mb: int,
                       log_dir: Path) -> None:
    """Query-tier switch: ALTER SYSTEM + restart to take effect."""
    psql_statement(cluster, f"ALTER SYSTEM SET shared_buffers = '{buffer_mb}MB'")
    subprocess.run(
        [str(pg_ctl(cluster)), "restart", "-D", str(cluster.pgdata),
         "-l", str(log_dir / "buffersize_restart.log"), "-w", "-t", "60"],
        check=True,
    )


def run_bench(
    args: argparse.Namespace, cluster: Cluster, label: str, commit: str,
    dataset: str, buffer_mb: int, output: Path, log_dir: Path,
    environment: dict[str, str],
) -> None:
    command = [
        sys.executable, str(BENCH_SCRIPT),
        "--dataset-name", dataset,
        "--variant", label, "--commit", commit,
        "--host", "localhost", "--port", str(cluster.port),
        "--dbname", DBNAME,
        "--data-dir", str(cluster.pgdata),
        "--pg-install", str(cluster.pg_install),
        "--pgctl-log-dir", str(log_dir / "pgctl"),
        "--m", str(M), "--ef-construction", str(EF_CONSTRUCTION),
        "--ef-search", str(EF_SEARCH),
        "--nqueries", str(NQUERIES),
        "--buffersize", f"{buffer_mb}MB",
        "--direct-io", "--drop-cache", "--skip-recall",
        "--output", str(output),
    ]
    run_stream(
        command, log_dir / f"{label}_{dataset}_{buffer_mb}MB.log",
        cwd=PROJECT_DIR, environment=environment,
    )
    if not valid_bench_csv(
        output, dataset, label, commit, buffer_mb, EF_SEARCH, NQUERIES,
    ):
        raise RuntimeError(f"result validation failed: {output}")


# ---------------------------------------------------------------------------
# Factor-level summary (idempotent rewrite, key = variant+dataset)
# ---------------------------------------------------------------------------

def rewrite_summary(work_dir: Path, datasets: list[str]) -> int:
    rows: dict[tuple[str, str], dict] = {}
    for label, _branch, _aftp in VERSIONS:
        combo_dir = work_dir / label
        if not combo_dir.exists():
            continue
        for csv_path in sorted(combo_dir.glob("*/read_bench_*.csv")):
            with csv_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    rows[(row["variant"], row["dataset"])] = row
    if not rows:
        return 0
    dataset_order = {name: i for i, name in enumerate(datasets)}
    variant_order = {label: i for i, (label, _, _) in enumerate(VERSIONS)}
    ordered = sorted(
        rows.values(),
        key=lambda row: (variant_order.get(row["variant"], 99),
                         dataset_order.get(row["dataset"], 99)),
    )
    output = work_dir / SUMMARY_NAME
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    print(f"summary written: {output} ({len(ordered)} rows)", flush=True)
    return len(ordered)


def run_version(
    args: argparse.Namespace, cluster: Cluster, label: str, commit: str,
    aftp: bool, work_dir: Path, log_dir: Path, page_size: int,
    buffer_map: dict[str, int] | None, environment: dict[str, str],
) -> dict[str, int]:
    """Single-variant leg: install -> start server -> build indexes (or reuse)
    -> query per dataset. Returns baseline's uniform tier map (other variants
    return the incoming argument unchanged)."""
    page_tag = f"{page_size // 1024}k"
    print(f"=== version {page_tag} {label} / {commit[:8]} ===", flush=True)
    install_version(args.repo_dir, cluster, log_dir,
                    f"{page_tag}_{label}", commit, environment)
    server_log = log_dir / f"{page_tag}_{label}_server.log"
    start_server(cluster, server_log)
    try:
        index_csv = index_csv_path(work_dir, label)
        datasets = list(args.datasets)

        # baseline: if the index CSV is already complete, read the tier map
        # first, then decide whether the whole variant can be skipped
        if label == "baseline" and index_csv_complete(index_csv, datasets):
            buffer_map = read_buffer_map(index_csv, datasets)

        bench_states = {
            name: valid_bench_csv(
                bench_csv_path(work_dir, label, commit, name,
                               buffer_map[name] if buffer_map else 0),
                name, label, commit,
                buffer_map[name] if buffer_map else 0,
                EF_SEARCH, NQUERIES,
            )
            for name in datasets
        }
        if args.resume and index_csv_complete(index_csv, datasets) \
                and all(bench_states.values()):
            print(f"SKIP version: {label} index and all result CSVs are complete.", flush=True)
        else:
            # Return to the uniform build-time config: the previous combo's
            # query tier lingers in ALTER SYSTEM, so 128MB must be restored
            # before rebuilding indexes (ordering as in util/16; the restart
            # reuses this variant's server_log so the AFTP fallback check
            # still reads the same file).
            psql_statement(
                cluster,
                f"ALTER SYSTEM SET shared_buffers = '{BUILD_SHARED_BUFFERS}'",
            )
            subprocess.run(
                [str(pg_ctl(cluster)), "restart", "-D", str(cluster.pgdata),
                 "-l", str(server_log), "-w", "-t", "60"],
                check=True,
            )
            for attempt in (1, 2):
                try:
                    build_indexes(args, cluster, label, commit, aftp,
                                  work_dir, log_dir, server_log, environment)
                    break
                except (subprocess.CalledProcessError, RuntimeError) as error:
                    if attempt == 2:
                        raise
                    print(f"warning: {label} index build failed ({error}), "
                          f"restarting the cluster and retrying once", flush=True)
                    stop_server(cluster)
                    start_server(cluster, server_log)

        if label == "baseline":
            buffer_map = read_buffer_map(index_csv, datasets)

        for name in datasets:
            buffer_mb = buffer_map[name]
            output = bench_csv_path(work_dir, label, commit, name, buffer_mb)
            if args.resume and valid_bench_csv(
                output, name, label, commit, buffer_mb, EF_SEARCH, NQUERIES,
            ):
                print(f"SKIP complete: {output}", flush=True)
                continue
            if output.exists():
                output.unlink()
            for attempt in (1, 2):
                try:
                    set_shared_buffers(cluster, buffer_mb, log_dir)
                    run_bench(args, cluster, label, commit, name, buffer_mb,
                              output, log_dir, environment)
                    break
                except (subprocess.CalledProcessError, RuntimeError) as error:
                    if attempt == 2:
                        raise
                    print(f"warning: {label}/{name} query failed ({error}), "
                          f"restarting the cluster and retrying once", flush=True)
                    stop_server(cluster)
                    start_server(cluster, server_log)
        return buffer_map
    finally:
        stop_server(cluster)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the main read-performance benchmark (5 variants × 4 datasets, single page size)."
    )
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--page-size", type=int, default=8192,
                        choices=PAGE_SIZE_ORDER)
    parser.add_argument("--repo-dir", type=Path,
                        default=PROJECT_DIR / "build" / "src")
    parser.add_argument("--variants", nargs="+",
                        default=[v[0] for v in VERSIONS],
                        choices=[v[0] for v in VERSIONS])
    parser.add_argument("--datasets", nargs="+", default=DATASET_ORDER,
                        choices=DATASET_ORDER)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-source", action="store_true",
                        help="do not restore the pre-experiment branch at the end (for debugging)")
    args = parser.parse_args()
    args.result_dir = args.result_dir.resolve()
    args.repo_dir = args.repo_dir.resolve()
    return args


def main() -> int:
    args = parse_args()
    page_size = args.page_size
    cluster = cluster_for(args.repo_dir, page_size)
    if not cluster.pgdata.exists():
        raise RuntimeError(f"target cluster data directory does not exist: {cluster.pgdata}")
    work_dir = args.result_dir / PAGE_SIZE_DIR[page_size]
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = command_environment(cluster)

    if git_status_lines(args.repo_dir):
        raise RuntimeError("repository working tree is dirty; commit or clean it before running.")
    versions = [(label, resolve_commit(args.repo_dir, branch), aftp)
                for label, branch, aftp in VERSIONS
                if label in args.variants]
    for label, commit, _aftp in versions:
        print(f"resolved {label}: {commit[:8]}")
    restore_ref = git_head_ref(args.repo_dir)
    was_running = pg_is_running(cluster)
    print(f"target cluster: pgdata={cluster.pgdata.name} port={cluster.port} "
          f"page_size={page_size} (running={was_running}), "
          f"restore ref={restore_ref}", flush=True)

    configure_server(cluster, work_dir, page_size)
    if pg_is_running(cluster):
        stop_server(cluster)

    page_tag = f"{page_size // 1024}k"
    buffer_map: dict[str, int] | None = None
    try:
        for label, commit, aftp in versions:
            if (page_size, label) in EXCLUDED_COMBOS:
                print(f"SKIP N/A: {label}@{page_tag} build is incompatible "
                      f"(established by Page Utilization); not run.", flush=True)
                continue
            # baseline must run first (it provides the uniform tier map)
            if buffer_map is None and label != "baseline":
                raise RuntimeError(
                    f"{label} needs the uniform tier map: run baseline first "
                    f"(--variants baseline or the full order).")
            buffer_map = run_version(
                args, cluster, label, commit, aftp, work_dir, log_dir,
                page_size, buffer_map, environment,
            )
            rewrite_summary(work_dir, args.datasets)
        rewrite_summary(work_dir, args.datasets)
    finally:
        if not args.keep_source:
            try:
                restore_repo(args.repo_dir, cluster, log_dir, restore_ref,
                             environment, log_prefix=page_tag)
                start_server(cluster,
                             log_dir / f"{page_tag}_restore_server.log")
                psql_statement(cluster, "ALTER SYSTEM RESET debug_io_direct")
                psql_statement(
                    cluster, "ALTER SYSTEM SET shared_buffers = '128MB'")
                subprocess.run(
                    [str(pg_ctl(cluster)), "restart", "-D", str(cluster.pgdata),
                     "-l", str(log_dir / f"{page_tag}_post_restore.log"),
                     "-w", "-t", "60"],
                    check=True,
                )
                if not was_running:
                    stop_server(cluster)
                    print("cluster left stopped, as before the experiment.", flush=True)
                else:
                    print("cluster config restored and left running.", flush=True)
            except Exception as error:  # noqa: BLE001 -- restore failures must be visible
                print(f"warning: failed to restore repository/cluster: {error}",
                      file=sys.stderr, flush=True)

    print(f"main read-performance benchmark ({page_size}B) complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

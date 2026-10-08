# GPO

## Overview
This repository provides an optimized implementation of pgvector with page-placement enhancements for high-performance disk-based HNSW vector search.
The optimizations focus on improving page utilization, boosting buffer hit rates, and preserving page-level locality, through Separation of Element and Neighbor (SEN) and Affinity-aware Tuple Layout (ATL).

## Branches
This repository consists of seven branches:

- **main**: README, benchmark scripts (`bench/`), the Deep100K dataset (`data/`), and the one-click runner (`run_all.sh`).
- **pg_orig**: Vanilla (original) PostgreSQL + pgvector.
- **Turbocharging**: Greedy graph partitioning + node colocation (prior work).
- **SEN**: Element–neighbor separation into type-homogeneous page chains.
- **ATL**: Co-access affinity from sampled queries + affinity-aware write ordering.
- **GPO**: SEN + ATL (type-homogeneous chains with affinity-aware write order).
- **motivation**: pgvector instrumented with per-layer read statistics and tuple-access tracing.

Branch topology mirrors the method structure (`git log --graph` on this repository):

```
* GPO         = merge(SEN, ATL)
|\
| *   SEN      (on pg_orig)
* |   ATL      (on pg_orig)
|/
| *   Turbocharging (on pg_orig)
|/
| *   motivation instrumentation (on pg_orig)
|/
*   pg_orig   vanilla PostgreSQL 17devel + pgvector v0.8.0
*   main     README + benchmark suite + Deep100K dataset  (root)
```

Every branch carries `bench/` and `data/`, so scripts and dataset stay
available on any checkout; code branches differ only in `postgres/` and
`pgvector/`.

| Paper method | Section | Branch |
|---|---|---|
| Baseline (pgvector 0.8.0) | §2 | `pg_orig` |
| Motivation instrumentation | §2 | `motivation` |
| TurboCharging (prior work) | §6 | `Turbocharging` |
| SEN | §3 | `SEN` |
| ATL | §4 | `ATL` |
| GPO (SEN + ATL) | §5 | `GPO` |

The benchmark driver resolves each branch name to its commit at runtime, so no
hashes are pinned anywhere in the scripts.

Note on naming: the ATL implementation in code uses the historical name **AFTP** (`pgvector/src/hnswaftp.c`, GUCs `hnsw.aftp_enable`, `hnsw.aftp_window`, `hnsw.aftp_sample_queries`, `hnsw.aftp_ef_search`). The GPO diff vs `pg_orig` is exactly the union of the SEN and ATL diffs.

## Prerequisites
To install the required dependencies, run:
```sh
apt install build-essential pkg-config flex bison libreadline-dev zlib1g-dev perl gdb rsync
```
Benchmarks additionally require Python 3.9+ with:
```sh
pip install -r requirements.txt
```

## Datasets
The **Deep100K** dataset ships with the repository under `data/`
(100,000 base vectors, 96-d, L2, plus 1,000 held-out queries with exact top-10
ground truth), originally from [Skoltech Deep1B](http://sites.skoltech.ru/compvision/noimi/).
Other evaluation datasets (SIFT / GloVe / NYTimes) can be staged in the same
layout — see `bench/README.md`.

## Repository Layout
```
GPO/
├── run_all.sh              # One-click: build all branches -> main experiment -> figures
├── bench/
│   ├── run_read_bench_main.py   # main-experiment orchestrator (5 versions, cold cache)
│   ├── measure_index_size.py    # index build + size measurement
│   ├── bench_read.py            # per-combination query benchmark
│   └── plots/                      # bar charts: fig07 QPS / fig08 hit rate / fig09 P95 / fig10 build time
├── data/
│   ├── Deep100K/           # 100K base vectors (Parquet)
│   └── queries/Deep100K/   # query-1000.parquet + ground-truth-k10.csv
├── results/                # output directory created by run_all.sh (empty in the repo)
└── postgres/, pgvector/    # only present on the six code branches
```

## How to Benchmark
To reproduce the main experiment (QPS, P95 latency, buffer hit rate, build
time; 5 versions × Deep100K, 8KB pages):

```sh
./run_all.sh                 # full pipeline: worktree -> postgres -> experiment -> figures
./run_all.sh --stage main    # a single stage: env | worktree | postgres | main | figures
./run_all.sh --dry-run       # print the planned commands without executing
```

`run_all.sh` uses `git worktree` to check out and build all compared versions in
`build/src`, so the working tree is never switched while scripts run. Outputs
land in `results/`: figures (PNG) in `results/figures/` and their source-data
CSVs in `results/csv/` (fig07–fig10, named after the paper). Intermediate
experiment outputs (`results/main/`) are removed automatically once the
figures have been rendered. The exact protocol (cold-cache procedure, buffer = 50% of the baseline
index size, HNSW parameters) is documented in `bench/README.md`.

Troubleshooting: the benchmark cluster listens on port 5432 — stop any other
PostgreSQL instance holding that port before running.

## Acknowledgments
This repository builds upon [PostgreSQL](https://github.com/postgres/postgres)
(17devel) and [pgvector](https://github.com/pgvector/pgvector) (v0.8.0), and
compares against the layout reorganization of TurboCharging.
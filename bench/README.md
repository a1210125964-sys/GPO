# Bench (minimal artifact)

Driver scripts for the main experiment and its figures.
All scripts are as-run ports with repo-relative paths; each docstring documents
its exact behavior and validation checks.

## One-click

```sh
./run_all.sh                          # worktree -> postgres(8KB) -> main experiment -> figures
GPO_DATASET=SIFT100K ./run_all.sh     # another dataset (must be staged under data/, see below)
```

## Scripts

| Script | Role |
|---|---|
| `run_read_bench_main.py` | Orchestrator: installs the five versions (pg_orig / Turbocharging / SEN / ATL / GPO) in `build/src`, rebuilds the index per version (`m=24, ef_construction=200`, ATL/GPO with `aftp_enable=on, aftp_window=32, aftp_sample_queries=400`), then runs the cold-cache query benchmark. Restores the worktree afterwards. `--resume` skips complete combinations. |
| `measure_index_size.py` | Loads the base vectors, runs `CREATE INDEX`, records index size and build memory (invoked by 27). |
| `bench_read.py` | Single-combination query benchmark: `shared_buffers` = 50% of the baseline index size, restart + drop cache + `debug_io_direct=data`, 1,000 queries under `EXPLAIN (ANALYZE, BUFFERS)` (invoked by 27). |
| `bench_read_common.py` | Dataset specs and Parquet loading. |
| `plots/fig07_09_main_results.py` | Grouped bars: `--metric {execution_qps, buffer_hit_rate, latency_p95_ms}` → paper Figs. 7/8/9. |
| `plots/fig10_buildtime.py` | Build-time bars → paper Fig. 10. |

Protocol summary: cold-cache single pass, ef_search=200, k=10, 8KB pages,
buffer tier fixed per dataset (baseline index size × 50%, shared by all five
versions).

## Datasets

`data/Deep100K` (100,000 base vectors, 96-d, L2) and `data/queries/Deep100K`
(1,000 held-out queries + exact top-10 ground truth) ship with the repository.

To evaluate another dataset (SIFT100K / Glove100K / NYTimes100K), stage the
same layout and pass `--datasets <name>`:

```
data/<Name100K>/build-0000{0,1,2}-of-00003.parquet   # base vectors (float32, "embedding" column)
data/<Name100K>/insert-00000-of-00001.parquet
data/queries/<Name100K>/query-1000.parquet           # 1,000 queries
data/queries/<Name100K>/ground-truth-k10.csv         # exact top-10
```

## Label equivalences

Historical internal labels that may appear in older CSVs map to the paper as:
`baseline`→Baseline, `TurboCharging`→TurboCharging, `Separation`/`SEN`→SEN,
`APL`/`AFTP`→ATL, `Separation+APL`→GPO. In code the ATL implementation keeps
the historical name **AFTP** (`hnswaftp.c`, GUCs `hnsw.aftp_*`).

#!/usr/bin/env bash
# GPO one-click benchmark pipeline (minimal artifact: main experiment on Deep100K).
#
#   ./run_all.sh                 # worktree -> postgres(8KB) -> main experiment -> figures
#   ./run_all.sh --stage main    # a single stage: env | worktree | postgres | main | figures
#   ./run_all.sh --dry-run       # print the planned commands
#
# The driver checks out and compiles the five compared versions inside the
# detached worktree build/src — the main working tree is never switched.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SRC="$REPO/build/src"
RESULTS="$REPO/results"
DATASET="${GPO_DATASET:-Deep100K}"
PY=python3

STAGE=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

run() {
    echo "+ $*"
    if [[ $DRY_RUN -eq 0 ]]; then "$@"; fi
}
run_in() {
    local cwd="$1"; shift
    echo "+ (cd $cwd) $*"
    if [[ $DRY_RUN -eq 0 ]]; then (cd "$cwd" && "$@"); fi
}

stage_env() {
    echo "== env: dependency check =="
    run $PY - <<'EOF'
import importlib.util, shutil, sys
missing = [m for m in ("numpy", "pyarrow", "psutil", "psycopg2", "matplotlib", "pandas")
           if importlib.util.find_spec(m) is None]
tools = [t for t in ("make", "flex", "bison", "git", "rsync") if shutil.which(t) is None]
if missing or tools:
    sys.exit(f"missing python modules: {missing}; missing tools: {tools}\n"
             "fix with: pip install -r requirements.txt  and  apt install <tools>")
print("environment OK")
EOF
}

stage_worktree() {
    echo "== worktree: prepare build/src =="
    if [[ -d $BUILD_SRC/.git || -f $BUILD_SRC/.git ]]; then
        echo "build/src already exists, reusing"
    else
        run git -C "$REPO" worktree add --detach "$BUILD_SRC" pg_orig
    fi
}

stage_postgres() {
    stage_worktree
    echo "== postgres: blocksize 8KB -> build/src/pg_install =="
    if [[ -x $BUILD_SRC/pg_install/bin/pg_ctl ]]; then
        echo "pg_install already built, skipping (delete it to force)"
    else
        run_in "$BUILD_SRC/postgres" ./configure --prefix="$BUILD_SRC/pg_install" \
            --with-blocksize=8 --without-icu --without-readline
        run_in "$BUILD_SRC/postgres" make -j"$(nproc)"
        run_in "$BUILD_SRC/postgres" make install
    fi
    stage_cluster
}

stage_cluster() {
    echo "== cluster: initdb + iotest database =="
    local pgdata="$BUILD_SRC/pgdata" bin="$BUILD_SRC/pg_install/bin" user
    user="$(id -un)"
    if [[ ! -f $pgdata/PG_VERSION ]]; then
        run "$bin/initdb" -D "$pgdata" -U "$user" --encoding=UTF8 --locale=C
    else
        echo "pgdata already initialized, reusing"
    fi
    if ! "$bin/pg_ctl" status -D "$pgdata" >/dev/null 2>&1; then
        run "$bin/pg_ctl" start -D "$pgdata" -l "$pgdata/bootstrap.log" -w -t 60
    fi
    if ! "$bin/psql" -h localhost -p 5432 -U "$user" -d postgres -tAc \
        "SELECT 1 FROM pg_database WHERE datname = 'iotest'" | grep -q 1; then
        run "$bin/createdb" -h localhost -p 5432 -U "$user" iotest
    else
        echo "database iotest already exists, reusing"
    fi
}

stage_main() {
    echo "== main experiment: 5 versions x $DATASET, 8KB pages =="
    stage_postgres
    run $PY "$REPO/bench/run_read_bench_main.py" \
        --result-dir "$RESULTS/main" --page-size 8192 \
        --datasets "$DATASET" --resume
}

stage_figures() {
    echo "== figures: QPS / buffer hit rate / P95 latency / build time bars =="
    mkdir -p "$RESULTS/figures" "$RESULTS/csv"
    for metric in execution_qps buffer_hit_rate latency_p95_ms; do
        run $PY "$REPO/bench/plots/fig07_09_main_results.py" --metric "$metric" \
            --datasets "$DATASET" --output-dir "$RESULTS/figures" --source-dir "$RESULTS/csv"
    done
    run $PY "$REPO/bench/plots/fig10_buildtime.py" \
        --datasets "$DATASET" --output-dir "$RESULTS/figures" --source-dir "$RESULTS/csv"
    echo "figures written to $RESULTS/figures, source CSVs to $RESULTS/csv"
}

stage_clean() {
    echo "== clean: remove intermediate experiment outputs =="
    run rm -rf "$RESULTS/main"
}

ALL_STAGES=(env worktree postgres main figures clean)
if [[ -z $STAGE ]]; then
    for s in "${ALL_STAGES[@]}"; do "stage_$s"; done
else
    "stage_$STAGE"
fi

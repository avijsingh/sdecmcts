#!/bin/bash
# Run all SDecMCTS benchmarks and collect results.
# Output goes to results/full_bench_TIMESTAMP.txt
#
# Labyrinth uses medium MCTS params (outer_iters=10, tau=20) to keep
# per-worker memory at ~3-4GB; default params use 20GB+ per worker.
# With --jobs 20 all 20 episodes run in parallel on the 20-core machine.

ROOT="$HOME/Documents/SDecZero/Scripts/sdecmcts"
OUT="$ROOT/results/full_bench_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$ROOT/results"

log() { echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT"; }

log "=== SDecMCTS Full Benchmark Suite ==="
log "Machine: $(hostname), CPUs: $(nproc)"
log ""

run_labyrinth() {
    local bench=$1 h=$2
    log "--- Labyrinth $bench H=$h ---"
    python3 -u -m benchmarks.labyrinth_online \
        --benchmark "$bench" --horizon "$h" \
        --episodes 50 --mode semi --jobs 20 \
        --outer-iters 10 --tau 20 --num-seq 10 --num-samples 10 \
        2>&1 | tee -a "$OUT" || log "WARN: $bench H=$h failed, continuing"
    log ""
}

run_tiger() {
    local h=$1
    log "--- Tiger H=$h ---"
    python3 -u -m benchmarks.semidec_tiger \
        --horizon "$h" --episodes 128 --seed 0 --max-tree-depth 1 \
        2>&1 | tee -a "$OUT"
    log ""
}

run_medevac() {
    local h=$1
    log "--- Medevac H=$h ---"
    python3 -u -m benchmarks.semidec_medevac \
        --horizon "$h" --episodes 128 --iterations 2000 --seed 0 --guide qmdp \
        2>&1 | tee -a "$OUT"
    log ""
}

run_mars() {
    local h=$1
    log "--- Mars H=$h ---"
    python3 -u -m benchmarks.semidec_mars \
        --horizon "$h" --episodes 128 --iterations 500 --seed 0 \
        --guide qmdp --max-tree-depth 1 --shared-belief \
        2>&1 | tee -a "$OUT"
    log ""
}

cd "$ROOT"

log "== Tiger =="
run_tiger 10
run_tiger 12

log "== Medevac =="
run_medevac 10

log "== Mars =="
run_mars 7
run_mars 10

log "== Labyrinth extcross9 =="
run_labyrinth extcross9 6
run_labyrinth extcross9 7
run_labyrinth extcross9 8

log "== Labyrinth lopsidedy10 =="
run_labyrinth lopsidedy10 5
run_labyrinth lopsidedy10 6
run_labyrinth lopsidedy10 7

log "== Labyrinth ladder10 =="
run_labyrinth ladder10 5
run_labyrinth ladder10 6
run_labyrinth ladder10 7

log "== Labyrinth maze12 =="
run_labyrinth maze12 5
run_labyrinth maze12 6
run_labyrinth maze12 7

log "== Labyrinth hiddentail11 =="
run_labyrinth hiddentail11 4
run_labyrinth hiddentail11 5
run_labyrinth hiddentail11 6

log "== Labyrinth mesh10 =="
run_labyrinth mesh10 5
run_labyrinth mesh10 6
run_labyrinth mesh10 7

log "=== All benchmarks complete ==="
log "Results saved to $OUT"

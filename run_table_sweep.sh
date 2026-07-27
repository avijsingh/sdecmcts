#!/bin/bash
# Run the full benchmark table sweep (all scenarios x horizons) and collect
# per-run results and wall times. Four streams run in parallel (tiger,
# medevac, mars, labyrinth); runs within a stream are sequential so each
# run's wall time is clean. Per-run timeout: 3600s.
#
# Usage: ./run_table_sweep.sh [outdir]

cd "$(dirname "$0")"
OUT="${1:-results/table_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

run() { # run <tag> <cmd...>
    local tag=$1; shift
    local log="$OUT/$tag.log"
    local t0 t1 rc
    t0=$(date +%s)
    timeout 3600 "$@" > "$log" 2>&1
    rc=$?
    t1=$(date +%s)
    if [ $rc -eq 124 ]; then
        echo "$tag: TIMEOUT (>3600s)" >> "$OUT/summary.txt"
    elif [ $rc -ne 0 ]; then
        echo "$tag: FAILED (rc=$rc, see $log)" >> "$OUT/summary.txt"
    else
        local res plan
        res=$(grep -E "^mean_return=" "$log" | tail -1)
        plan=$(grep -E "plan_time/episode" "$log" | tail -1)
        echo "$tag: $res ${plan:+| $plan }| wall=$((t1 - t0))s" >> "$OUT/summary.txt"
    fi
}

tiger_stream() {
    for h in 10 12 15 20; do
        run "tiger_h$h" python3 -u -m benchmarks.semidec_tiger \
            --horizon "$h" --episodes 128 --seed 0 --max-tree-depth 1
    done
}

medevac_stream() {
    for h in 7 8 9 10; do
        run "medevac_h$h" python3 -u -m benchmarks.semidec_medevac \
            --horizon "$h" --episodes 128 --iterations 2000 --seed 0 --guide qmdp
    done
}

mars_stream() {
    for h in 7 8 9 10; do
        run "mars_h$h" python3 -u -m benchmarks.semidec_mars \
            --horizon "$h" --episodes 128 --iterations 500 --seed 0 \
            --guide qmdp --max-tree-depth 1 --shared-belief
    done
}

labyrinth_stream() {
    local grid="extcross9:6,7,8 lopsidedy10:5,6,7 ladder10:5,6,7 \
maze12:5,6,7 hiddentail11:4,5,6 mesh10:5,6,7"
    for entry in $grid; do
        local bench=${entry%%:*}
        for h in $(echo "${entry#*:}" | tr , ' '); do
            run "labyrinth_${bench}_h$h" python3 -u -m benchmarks.semidec_labyrinth \
                --benchmark "$bench" --horizon "$h" --episodes 2 --jobs 2 --seed 0
        done
    done
}

tiger_stream &
medevac_stream &
mars_stream &
labyrinth_stream &
wait

sort "$OUT/summary.txt" > "$OUT/summary_sorted.txt"
echo "=== SWEEP COMPLETE $(date '+%F %T') ===" >> "$OUT/summary.txt"
echo "Results in $OUT"

#!/usr/bin/env bash
# ======================================================================
#  run_all_experiments.sh — Full Experiment Suite (Sections 1–15)
#
#  Runs ALL 6 sweep types in parallel, each with:
#    - 30 seeds per condition (--runs 30)
#    - 5 generations (--generations 5)
#    - Internal multiprocessing (--workers 0 = all cores)
#    - Full analysis pipeline: batch logs + stats + figures (--analyze)
#
#  Output structure:
#    data/results/
#      ├── selection_criteria/   (4 conditions × 30 = 120 runs)
#      ├── ratio_sweep/          (4 conditions × 30 = 120 runs)
#      ├── duration_sweep/       (4 conditions × 30 = 120 runs)
#      ├── resource_variation/   (3 conditions × 30 =  90 runs)
#      ├── no_return/            (2 conditions × 30 =  60 runs)
#      └── generation_length/    (4 conditions × 30 = 120 runs)
#                                                 Total: 630 runs
#
#  Usage:
#    chmod +x scripts/run_all_experiments.sh
#    ./scripts/run_all_experiments.sh                    # default: 30 runs, 5 gens
#    ./scripts/run_all_experiments.sh --runs 10 --gens 3 # custom
#    ./scripts/run_all_experiments.sh --serial            # one at a time (low RAM)
# ======================================================================

set -e

# Defaults
RUNS=30
GENS=5
WORKERS=0
OUTDIR="data/results"
SERIAL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --runs)    RUNS="$2";    shift 2 ;;
    --gens)    GENS="$2";    shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --outdir)  OUTDIR="$2";  shift 2 ;;
    --serial)  SERIAL=true;  shift   ;;
    -h|--help)
      echo "Usage: $0 [--runs N] [--gens N] [--workers N] [--outdir DIR] [--serial]"
      echo "  --runs N      Seeds per condition (default: 30)"
      echo "  --gens N      Generations (default: 5)"
      echo "  --workers N   CPU cores per sweep (default: 0 = all)"
      echo "  --outdir DIR  Output directory (default: data/results)"
      echo "  --serial      Run sweeps sequentially (saves RAM)"
      exit 0 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

# All 6 experiment types
SWEEPS=(
  "selection_criteria"
  "ratio_sweep"
  "duration_sweep"
  "resource_variation"
  "no_return"
  "generation_length"
)

# Condition counts for display
declare -A CONDS
CONDS[selection_criteria]=4
CONDS[ratio_sweep]=4
CONDS[duration_sweep]=4
CONDS[resource_variation]=3
CONDS[no_return]=2
CONDS[generation_length]=4

TOTAL_CONDS=0
for s in "${SWEEPS[@]}"; do
  TOTAL_CONDS=$((TOTAL_CONDS + CONDS[$s]))
done
TOTAL_RUNS=$((TOTAL_CONDS * RUNS))

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     SWARM ISOLATION SIMULATION — FULL EXPERIMENT SUITE      ║"
echo "║                    Sections 1–15                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Sweeps:       ${#SWEEPS[@]} experiment types                          ║"
printf "║  Conditions:   %-2d total                                     ║\n" "$TOTAL_CONDS"
printf "║  Seeds:        %-2d per condition                             ║\n" "$RUNS"
printf "║  Total runs:   %-4d                                         ║\n" "$TOTAL_RUNS"
printf "║  Generations:  %-2d                                           ║\n" "$GENS"
printf "║  Workers:      %-4s per sweep                               ║\n" "$( [ $WORKERS -eq 0 ] && echo 'ALL' || echo $WORKERS )"
printf "║  Mode:         %-10s                                    ║\n" "$( $SERIAL && echo 'SERIAL' || echo 'PARALLEL' )"
printf "║  Output:       %-40s  ║\n" "$OUTDIR"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$OUTDIR"
START=$(date +%s)

# Build per-sweep command
run_sweep() {
  local sweep=$1
  local sweep_dir="$OUTDIR/$sweep"
  local log="$OUTDIR/${sweep}.log"
  local nconds=${CONDS[$sweep]}
  local nruns=$((nconds * RUNS))

  echo "[$(date +%H:%M:%S)] ▶ Starting $sweep ($nconds conditions × $RUNS seeds = $nruns runs)"

  python3 scripts/run_simulation.py \
    --sweep "$sweep" \
    --generations "$GENS" \
    --runs "$RUNS" \
    --workers "$WORKERS" \
    --analyze \
    --plot "$sweep_dir" \
    > "$log" 2>&1

  local rc=$?
  local elapsed=$(($(date +%s) - START))
  if [ $rc -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] ✅ $sweep completed (${elapsed}s elapsed)"
  else
    echo "[$(date +%H:%M:%S)] ❌ $sweep FAILED (exit $rc, see $log)"
  fi
  return $rc
}

# Execute
PIDS=()
NAMES=()

if $SERIAL; then
  echo "[*] Running sweeps sequentially..."
  echo ""
  for sweep in "${SWEEPS[@]}"; do
    run_sweep "$sweep"
  done
else
  echo "[*] Launching all ${#SWEEPS[@]} sweeps in parallel..."
  echo ""
  for sweep in "${SWEEPS[@]}"; do
    run_sweep "$sweep" &
    PIDS+=($!)
    NAMES+=("$sweep")
  done

  # Wait for all and collect results
  FAILED=0
  for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || {
      echo "  ❌ ${NAMES[$i]} failed"
      FAILED=$((FAILED + 1))
    }
  done
fi

END=$(date +%s)
DURATION=$((END - START))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ALL EXPERIMENTS COMPLETE                  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Total time:   %dm %02ds                                     ║\n" "$MINUTES" "$SECONDS"
printf "║  Total runs:   %-4d                                         ║\n" "$TOTAL_RUNS"
echo "║  Output:       $OUTDIR/"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Summary of output files
echo "Output files:"
for sweep in "${SWEEPS[@]}"; do
  local_dir="$OUTDIR/$sweep"
  if [ -d "$local_dir" ]; then
    nfiles=$(find "$local_dir" -type f | wc -l)
    sz=$(du -sh "$local_dir" | cut -f1)
    echo "  $sweep/: $nfiles files ($sz)"
  else
    echo "  $sweep/: NOT FOUND"
  fi
done

echo ""
echo "Per-sweep logs: $OUTDIR/*.log"
echo ""
echo "Done!"

#!/usr/bin/env bash
# Run the full experiment pipeline end-to-end.
#
# Usage:
#   scripts/reproduce_results.sh                  # cloud production run
#   CONFIG=configs/local-pilot.yaml scripts/reproduce_results.sh
#
# Required environment for the cloud run:
#   TOGETHER_API_KEY=<your Together AI key>

set -euo pipefail

CONFIG="${CONFIG:-configs/cloud-production.yaml}"
RESULTS="${RESULTS:-results}"

echo "Experiment pipeline starting"
echo "  config:       $CONFIG"
echo "  results dir:  $RESULTS"
echo

llm-memory-eval download-data
llm-memory-eval prepare-data
llm-memory-eval run --config "$CONFIG" --output "$RESULTS"
llm-memory-eval analyze --results-dir "$RESULTS"
llm-memory-eval figures --results-dir "$RESULTS"

echo
echo "Experiment pipeline complete."
echo "Outputs:"
echo "  $RESULTS/experiment_results.csv"
echo "  $RESULTS/statistical_analyses.json"
echo "  $RESULTS/tables/*.csv"
echo "  $RESULTS/figures/*.png"

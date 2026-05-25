#!/usr/bin/env bash
# Reproduce every result in the manuscript end-to-end.
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
DISSERTATION="${DISSERTATION:-dissertation}"

echo "Reproduction pipeline starting"
echo "  config:       $CONFIG"
echo "  results dir:  $RESULTS"
echo "  output dir:   $DISSERTATION"
echo

llm-memory-eval download-data
llm-memory-eval prepare-data
llm-memory-eval run --config "$CONFIG" --output "$RESULTS"
llm-memory-eval analyze --results-dir "$RESULTS"
llm-memory-eval figures --results-dir "$RESULTS"
llm-memory-eval build-chapters --results-dir "$RESULTS" --output "$DISSERTATION"

echo
echo "Reproduction pipeline complete."
echo "Outputs:"
echo "  $RESULTS/experiment_results.csv"
echo "  $RESULTS/statistical_analyses.json"
echo "  $RESULTS/tables/*.csv"
echo "  $RESULTS/figures/*.png"
echo "  $DISSERTATION/Chapter_4.docx"
echo "  $DISSERTATION/Chapter_5.docx"

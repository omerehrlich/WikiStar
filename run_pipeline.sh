#!/usr/bin/env bash
# Wrapper around the pipeline orchestrator (wiki_pipeline/pipeline.py).
#
# Usage:
#   ./run_pipeline.sh <article_name> <output_csv_path> [model_name]
#
# Example:
#   ./run_pipeline.sh "CRISPR" "results/CRISPR_classified.csv" "gpt-5-mini"
#
# Before running, export your OpenAI key:
#   export OPENAI_API_KEY=your_api_key_here

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <article_name> <output_csv_path> [model_name]" >&2
    exit 1
fi

ARTICLE="$1"
OUTPUT="$2"
MODEL="${3:-gpt-5-mini}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set." >&2
    echo "Run: export OPENAI_API_KEY=your_api_key_here" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run from the repo root so the `wiki_pipeline` package is importable.
cd "$SCRIPT_DIR"
python -m wiki_pipeline.pipeline \
    --article "$ARTICLE" \
    --output "$OUTPUT" \
    --model "$MODEL"

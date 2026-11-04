#!/usr/bin/env bash
set -euo pipefail
output=${1:-dist}
mkdir -p "$output"
python -m build --wheel --outdir "$output"
cp uv.lock "$output/uv.lock"

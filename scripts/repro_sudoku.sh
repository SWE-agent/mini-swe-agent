#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m minisweagent.run.sudoku --difficulty easy <<'INP'
show
hint
set 1,3 4
clear 1,3
quit
INP

#!/usr/bin/env bash
# Colleague's call graph -> our R1-R3 marks on the SAME graph (only what goes through the colleague's wrappers)
# -> wrapper targets + CFGs + click-through HTML.
#
# usage:
#   bash <verification folder>/run_pipeline.sh [preset] [REPO_DIR] [OUT_DIR]
#     preset    default: mini (this copy lives in that repo; presets: mini | smolagents | browser-use)
#     REPO_DIR  default: the repo this script is in
#     OUT_DIR   default: <this verification folder>/out_pipeline
#   Every tool it uses (audit.py, llm_rules.py, mark_colleague.py, audit_rules.py, wrapper_targets.py,
#   cfg_builder.py, report.py) sits next to this script.
#
#   1_colleague_callgraph/   audit.py exactly as the colleague ran it (same scope, entry, CFG filter)
#                            + callgraph_*_marked.png/svg: the SAME graph, red = entry -> colleague's wrapper ->
#                              LLM call; the LLM-calling functions outside the scope hang off it (red dashed box)
#                              marked_nodes.tsv: why each node is coloured
#   2_rules_callgraph/       (mini only; RULES_STEP=1 to force) audit_rules.py: whole package, all CFG targets
#   3_wrapper_targets/       wrapper_targets.py --through <colleague's wrappers>: wrap targets, coverage verdict,
#                            CFGs with exception edges, report.html (click a target -> its CFG)
#
# Works with macOS's bash 3.2.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PRESET="${1:-mini}"
ONLY=(); EXCLUDE=(); CFG_FILTER=()
case "$PRESET" in
  mini)
    VDIR="PoC/verification"
    PKG="src/minisweagent"; AGENTS="src/minisweagent/agents"; EXCLUDE=(InteractiveAgent)
    ENTRY="DefaultAgent.run"; RULES_STEP="${RULES_STEP:-1}"
    THROUGH=(agents/default.py::DefaultAgent.query) ;;
  smolagents)
    VDIR="verification"
    PKG="src/smolagents"; AGENTS="src/smolagents"; ONLY=(agents.py tools.py)
    ENTRY="MultiStepAgent.run"; CFG_FILTER=(MultiStepAgent CodeAgent ToolCallingAgent BaseTool Tool.__call__)
    RULES_STEP="${RULES_STEP:-0}"
    THROUGH=(agents.py::MultiStepAgent._generate_planning_step agents.py::CodeAgent._step_stream
             agents.py::ToolCallingAgent._step_stream agents.py::MultiStepAgent.provide_final_answer) ;;
  browser-use)
    VDIR="PoC/verification"
    PKG="browser_use"; AGENTS="browser_use"; ONLY=(agent/service.py tools/registry/service.py)
    ENTRY="Agent.run"; RULES_STEP="${RULES_STEP:-0}"
    THROUGH=(agent/service.py::Agent.get_model_output) ;;
  *) echo "unknown preset: $PRESET (mini | smolagents | browser-use)"; exit 1 ;;
esac
UP="$(echo "$VDIR" | sed 's#[^/][^/]*#..#g')"   # PoC/verification -> ../..  ;  verification -> ..
REPO="$(cd "${2:-$HERE/$UP}" && pwd)"
OUT="${3:-$REPO/$VDIR/out_pipeline}"
# Python for the analysis (independent of the repo's own pin): UV_PYTHON if set, otherwise an ALREADY INSTALLED
# Python >= 3.10, so uv never has to download an interpreter (the repo's .python-version is ignored).
if [ -z "${UV_PYTHON:-}" ]; then
  UV_PYTHON="$(cd / && uv python find --no-python-downloads '>=3.10' 2>/dev/null || true)"
  if [ -z "$UV_PYTHON" ]; then echo "no installed Python >= 3.10 found: install one (uv python install 3.12) or set UV_PYTHON"; exit 1; fi
fi
export UV_PYTHON
py() {  # online first; if the network is down, retry once with the packages already in uv's cache
  uv run --no-project --no-python-downloads --with py2cfg --with graphviz --with astor python "$@" \
    || { echo "-- retrying offline with cached packages"; \
         uv run --offline --no-project --no-python-downloads --with py2cfg --with graphviz --with astor python "$@"; }
}

O_ONLY=();  if [ ${#ONLY[@]} -gt 0 ];       then O_ONLY=(--only "${ONLY[@]}"); fi
O_EXCL=();  if [ ${#EXCLUDE[@]} -gt 0 ];    then O_EXCL=(--exclude "${EXCLUDE[@]}"); fi
O_CFG=();   if [ ${#CFG_FILTER[@]} -gt 0 ]; then O_CFG=(--cfg "${CFG_FILTER[@]}"); fi

cd "$REPO"
mkdir -p "$OUT"
echo "== [$PRESET] $REPO -> $OUT"

echo "== 1 colleague: audit.py ($AGENTS) + our marks on the same graph"
py "$HERE/audit.py" "$AGENTS" "$OUT/1_colleague_callgraph" --entry "$ENTRY" \
  ${O_ONLY[@]+"${O_ONLY[@]}"} ${O_EXCL[@]+"${O_EXCL[@]}"} ${O_CFG[@]+"${O_CFG[@]}"}
py "$HERE/mark_colleague.py" "$PKG" "$AGENTS" "$OUT/1_colleague_callgraph" --entry "$ENTRY" --through auto \
  ${O_ONLY[@]+"${O_ONLY[@]}"} ${O_EXCL[@]+"${O_EXCL[@]}"}

if [ "$RULES_STEP" = "1" ]; then
  echo "== 2 ours: audit_rules.py ($PKG)"
  py "$HERE/audit_rules.py" "$PKG" "$OUT/2_rules_callgraph" --entry "$ENTRY" --view-max-links 1 --cfg
fi

echo "== 3 ours: wrapper_targets.py ($PKG, through the colleague's wrappers)"
py "$HERE/wrapper_targets.py" "$PKG" "$OUT/3_wrapper_targets" --through "${THROUGH[@]}" --cfg

echo
echo "done -> $OUT"
echo "  colleague's graph with red marks: 1_colleague_callgraph/callgraph_entry_marked.png"
echo "  click-through report:             3_wrapper_targets/report.html"

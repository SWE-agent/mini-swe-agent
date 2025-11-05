#!/bin/bash
# Master script to run full A-mem integration experiment
# This script:
# 1. Sets up environment
# 2. Runs baseline experiment
# 3. Runs memory-enhanced experiment
# 4. Compares results

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "======================================================================="
echo "  A-mem Integration Experiment for Mini-SWE-Agent"
echo "======================================================================="
echo -e "${NC}"

# Configuration
WORK_DIR="/common/users/wx139/code/mini-swe-agent"
ENV_PATH="/common/users/wx139/env/minisweagentenv"
RESULTS_DIR="${WORK_DIR}/experiments/results"

# Parse arguments
RUN_BASELINE=true
RUN_MEMORY=true
RUN_ANALYSIS=true
SKIP_SETUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-baseline)
            RUN_BASELINE=false
            shift
            ;;
        --skip-memory)
            RUN_MEMORY=false
            shift
            ;;
        --skip-analysis)
            RUN_ANALYSIS=false
            shift
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --baseline-only)
            RUN_BASELINE=true
            RUN_MEMORY=false
            RUN_ANALYSIS=false
            shift
            ;;
        --memory-only)
            RUN_BASELINE=false
            RUN_MEMORY=true
            RUN_ANALYSIS=false
            shift
            ;;
        --analysis-only)
            RUN_BASELINE=false
            RUN_MEMORY=false
            RUN_ANALYSIS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-baseline      Skip baseline experiment"
            echo "  --skip-memory        Skip memory experiment"
            echo "  --skip-analysis      Skip analysis"
            echo "  --skip-setup         Skip environment setup"
            echo "  --baseline-only      Run only baseline experiment"
            echo "  --memory-only        Run only memory experiment"
            echo "  --analysis-only      Run only analysis"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Step 0: Environment setup
if [ "$SKIP_SETUP" = false ]; then
    echo -e "${YELLOW}[Step 0/4] Setting up environment...${NC}"

    # Activate environment
    source ${ENV_PATH}/bin/activate

    # Set environment variables
    export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"

    # Check for API key
    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${RED}Error: OPENAI_API_KEY not set${NC}"
        echo "Please set your OpenAI API key before running:"
        echo "  export OPENAI_API_KEY='your-key-here'"
        exit 1
    fi

    echo -e "${GREEN}✓ Environment ready${NC}"
    echo "  Python: $(which python)"
    echo "  Working directory: ${WORK_DIR}"
    echo ""
else
    echo -e "${YELLOW}[Step 0/4] Skipping setup (--skip-setup flag)${NC}"
    echo ""
fi

# Step 1: Run baseline experiment
if [ "$RUN_BASELINE" = true ]; then
    echo -e "${YELLOW}[Step 1/4] Running baseline experiment...${NC}"
    echo "  This may take several minutes depending on dataset size."
    echo ""

    cd ${WORK_DIR}
    bash experiments/run_baseline.sh

    echo ""
    echo -e "${GREEN}✓ Baseline experiment completed${NC}"
    echo ""
else
    echo -e "${YELLOW}[Step 1/4] Skipping baseline experiment${NC}"
    echo ""
fi

# Step 2: Run memory-enhanced experiment
if [ "$RUN_MEMORY" = true ]; then
    echo -e "${YELLOW}[Step 2/4] Running memory-enhanced experiment...${NC}"
    echo "  This may take several minutes depending on dataset size."
    echo ""

    cd ${WORK_DIR}
    bash experiments/run_memory.sh

    echo ""
    echo -e "${GREEN}✓ Memory experiment completed${NC}"
    echo ""
else
    echo -e "${YELLOW}[Step 2/4] Skipping memory experiment${NC}"
    echo ""
fi

# Step 3: Analyze and compare results
if [ "$RUN_ANALYSIS" = true ]; then
    echo -e "${YELLOW}[Step 3/4] Analyzing results...${NC}"
    echo ""

    cd ${WORK_DIR}
    python experiments/analyze_results.py \
        --baseline-dir ${RESULTS_DIR}/baseline \
        --memory-dir ${RESULTS_DIR}/memory \
        --output ${RESULTS_DIR}/comparison.json

    echo ""
    echo -e "${GREEN}✓ Analysis completed${NC}"
    echo ""
else
    echo -e "${YELLOW}[Step 3/4] Skipping analysis${NC}"
    echo ""
fi

# Step 4: Summary
echo -e "${BLUE}"
echo "======================================================================="
echo "  Experiment Complete!"
echo "======================================================================="
echo -e "${NC}"
echo ""
echo "📂 Results Location:"
echo "  Baseline:   ${RESULTS_DIR}/baseline/"
echo "  Memory:     ${RESULTS_DIR}/memory/"
echo "  Comparison: ${RESULTS_DIR}/comparison.json"
echo ""
echo "📊 Next Steps:"
echo "  1. Review comparison.json for detailed metrics"
echo "  2. Check individual trajectories in results/*/trajectories/"
echo "  3. Examine logs: results/*/run.log"
echo "  4. Inspect memory database: experiments/memory_db/"
echo ""
echo "🔄 To re-run analysis only:"
echo "  ./run_full_experiment.sh --analysis-only"
echo ""
echo -e "${GREEN}All done! 🎉${NC}"

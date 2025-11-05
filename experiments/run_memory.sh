#!/bin/bash
# Run memory-enhanced experiment with A-mem

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running Memory-Enhanced Experiment${NC}"
echo -e "${BLUE}========================================${NC}"

# Activate environment
echo -e "${GREEN}Activating environment...${NC}"
source /common/users/wx139/env/minisweagentenv/bin/activate

# Set working directory
cd /common/users/wx139/code/mini-swe-agent

# Create output directories
mkdir -p experiments/results/memory
mkdir -p experiments/memory_db

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:/common/users/wx139/code/mini-swe-agent/src"
export OPENAI_API_KEY="${OPENAI_API_KEY}"  # Should be set in environment

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY not set${NC}"
    echo "Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

echo -e "${GREEN}Starting memory-enhanced experiment...${NC}"
echo "Config: experiments/configs/memory_config.yaml"
echo "Output: experiments/results/memory/"
echo "Memory DB: experiments/memory_db/"
echo ""

# Run the experiment
python -m minisweagent.run \
    --config experiments/configs/memory_config.yaml \
    --output-dir experiments/results/memory \
    2>&1 | tee experiments/results/memory/run.log

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Memory experiment completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Results saved to: experiments/results/memory/"
echo "Memory database: experiments/memory_db/"
echo "Log file: experiments/results/memory/run.log"

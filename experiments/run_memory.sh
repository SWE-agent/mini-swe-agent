#!/bin/bash
# Run memory experiment using test_memagent.py

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running Memory Experiment (with A-mem)${NC}"
echo -e "${BLUE}========================================${NC}"

# Activate environment
echo -e "${GREEN}Activating environment...${NC}"
source /common/users/wx139/env/minisweagentenv/bin/activate

# Set working directory
cd /common/users/wx139/code/mini-swe-agent

# Create output directories
mkdir -p experiments/results/memory

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY not set${NC}"
    echo "Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

echo -e "${GREEN}Starting memory experiment with MemoryAgent...${NC}"
echo "Config: experiments/configs/memory_swebench.yaml"
echo "Output: experiments/results/memory/"
echo ""

# Run the memory experiment using test_memagent.py
python src/minisweagent/run/extra/test_memagent.py \
  --output experiments/results/memory \
  --config experiments/configs/memory_swebench.yaml \
  --model openai/gpt-5-nano-2025-08-07 \
  --environment-class swerex_docker \
  --subset lite \
  --slice "0:5" \
  --workers 1

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Memory experiment completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Results saved to: experiments/results/memory/"
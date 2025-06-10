#!/bin/bash

# =============================================================================
# LLM Poker Solver - Main Launcher
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    LLM Poker Solver                         ║"
echo "║              Advanced Poker AI Analysis Tool                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}❌ Conda not found. Please install Anaconda or Miniconda first.${NC}"
        echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
}

# Function to check if environment exists
check_environment() {
    if ! conda env list | grep -q "^poker-llm "; then
        echo -e "${YELLOW}⚠️  Environment 'poker-llm' not found.${NC}"
        echo -e "${BLUE}🔧 Creating environment from environment.yaml...${NC}"
        conda env create -f environment.yaml
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Environment created successfully!${NC}"
        else
            echo -e "${RED}❌ Failed to create environment. Please check environment.yaml${NC}"
            exit 1
        fi
    fi
}

# Function to activate environment
activate_environment() {
    echo -e "${BLUE}🔄 Activating poker-llm environment...${NC}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate poker-llm
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Environment activated successfully${NC}"
    else
        echo -e "${RED}❌ Failed to activate environment${NC}"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 [COMMAND]"
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo "  gui                    Launch the graphical user interface (default)"
    echo "  evaluate               Run evaluation scripts"
    echo "  train                  Run training scripts"
    echo "  solve                  Run solver scripts"
    echo "  help                   Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0                     # Launch GUI"
    echo "  $0 gui                 # Launch GUI"
    echo "  $0 evaluate            # Run evaluation"
    echo ""
}

# Function to launch GUI
launch_gui() {
    echo -e "${BLUE}🚀 Starting SystemC Poker Solver GUI...${NC}"
    cd scripts/app
    python poker_gui.py
}

# Function to run evaluation
run_evaluation() {
    echo -e "${BLUE}🔍 Running evaluation scripts...${NC}"
    cd scripts/evaluation
    echo "Available evaluation scripts:"
    ls -1 *.py 2>/dev/null | head -5
    echo ""
    echo "Please run specific evaluation scripts manually from scripts/evaluation/"
}

# Function to run training
run_training() {
    echo -e "${BLUE}🏋️  Training scripts available...${NC}"
    cd scripts/training_data
    echo "Available training directories:"
    ls -d */ 2>/dev/null
    echo ""
    echo "Please run specific training scripts manually from scripts/training_data/"
}

# Function to run solver
run_solver() {
    echo -e "${BLUE}🧠 Solver scripts available...${NC}"
    cd scripts/solver
    echo "Available solver scripts:"
    ls -1 *.py 2>/dev/null | head -5
    echo ""
    echo "Please run specific solver scripts manually from scripts/solver/"
}

# Main execution
main() {
    local command=${1:-gui}
    
    case $command in
        "gui" | "")
            check_conda
            check_environment
            activate_environment
            launch_gui
            ;;
        "evaluate")
            check_conda
            check_environment
            activate_environment
            run_evaluation
            ;;
        "train")
            check_conda
            check_environment
            activate_environment
            run_training
            ;;
        "solve")
            check_conda
            check_environment
            activate_environment
            run_solver
            ;;
        "help" | "-h" | "--help")
            show_usage
            ;;
        *)
            echo -e "${RED}❌ Unknown command: $command${NC}"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Check if we're in the right directory
if [[ ! -f "environment.yaml" ]]; then
    echo -e "${RED}❌ Please run this script from the project root directory${NC}"
    echo "   (The directory containing environment.yaml)"
    exit 1
fi

# Run main function with all arguments
main "$@" 
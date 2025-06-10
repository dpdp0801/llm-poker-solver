#!/bin/bash

echo "🚀 Starting SystemC Poker Solver GUI..."

# Source conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate environment and run GUI
conda activate poker-llm
if [ $? -eq 0 ]; then
    echo "✅ Environment activated successfully"
    python poker_gui.py
else
    echo "❌ Failed to activate poker-llm environment"
    echo "Please run: conda activate poker-llm && python poker_gui.py"
fi

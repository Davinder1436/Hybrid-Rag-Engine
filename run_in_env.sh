#!/bin/bash
# Helper script to run commands in the 'research' conda environment
# Usage: ./run_in_env.sh <command>

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found."
    exit 1
fi

# Activate research environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate research

# Show environment info
echo "✅ Environment: $CONDA_DEFAULT_ENV"
echo "✅ Python version: $(python --version)"
echo ""

# Run the provided command
if [ $# -eq 0 ]; then
    # No arguments - just activate and start a shell
    echo "Starting shell in 'research' environment..."
    exec $SHELL
else
    # Run the command
    exec "$@"
fi

#!/bin/bash

# Ensure the script uses the virtual environment
if [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python3"
    echo "Warning: .venv not found. Falling back to system python3."
fi

echo "Generating all individual subpanels (no text, colorbars, or annotations)..."
$PYTHON_CMD src/generate_subpanels.py
echo "Subpanel generation complete."

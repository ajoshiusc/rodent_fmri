#!/bin/bash

# ==============================================================================
# Generate All Figures Script
#
# This script sequentially runs all the Python figure generation codes 
# to recreate Figure 1-5 and Supplemental Figures S1-S4 for the paper.
# ==============================================================================

# Ensure the script uses the virtual environment where nilearn, matplotlib, etc. are installed
if [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python3"
    echo "Warning: .venv not found. Falling back to system python3."
fi

# Store the output directory for figures, default to current directory
OUTDIR="${1:-.}"

echo "=========================================="
echo "    Starting Figure Generation Pipeline   "
echo "=========================================="

echo "[1/9] Generating Figure 1 (BrainSync functional distances)..."
$PYTHON_CMD src/generate_figure1_brainsync_distances.py 

echo "[2/9] Generating Figure 2 (Effect size maps)..."
$PYTHON_CMD src/generate_figure2_effect_size.py 

echo "[3/9] Generating Figure 3 (Statistical power maps)..."
$PYTHON_CMD src/generate_figure3_power.py 

echo "[4/9] Generating Figure 4 (ROI FDR-corrected p-value thresholds)..."
$PYTHON_CMD src/generate_figure4_pvals.py 

echo "[5/9] Generating Figure 5 (BrainSync detailed evaluation - uncorrected)..."
$PYTHON_CMD src/generate_figure5_brainsync.py 

echo "[6/9] Generating Figure S1 (Connectivity profile distances)..."
$PYTHON_CMD src/generate_figureS1_conn_distances.py 

echo "[7/9] Generating Figure S2 (Node degree distances)..."
$PYTHON_CMD src/generate_figureS2_degree_distances.py 

echo "[8/9] Generating Figure S3 (Connectivity profile p-values)..."
$PYTHON_CMD src/generate_figureS3_conn_pvals.py 

echo "[9/9] Generating Figure S4 (Node degree p-values)..."
$PYTHON_CMD src/generate_figureS4_degree_pvals.py 

echo "=========================================="
echo "    All figures generated successfully!   "
echo "=========================================="

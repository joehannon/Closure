#!/bin/bash
# Regenerate results, epsilon sweeps, movies, and PDFs for a fixed set of inputs.
# Run with: ./regenerate_all.sh
set -e

INPUTS=(
    input_acetal_hydrolysis_lindenberg_lower
    input_ester_hydrolysis
    input_Dushman
    input_fuller_BC_azo_coupling_kinetics
    input_2025_paper_kinetics
    input_original_azo_coupling
    input_one_rxn
)

for name in "${INPUTS[@]}"; do
    echo "=== $name ==="
    python3 species_limits.py "inputs/${name}.json" --sweep --movie --pdf
done

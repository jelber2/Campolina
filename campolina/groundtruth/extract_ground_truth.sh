#!/bin/bash

# Usage:
# ./extract_ground_truth.sh <dorado_path> <GPU> <reference.fasta> <model> <reads.pod5> <level_table> <save_bam>

set -e  # Exit on error

# Input arguments
DORADO="$1"
GPU="$2"
REFERENCE="$3"
MODEL="$4"
READS="$5"
LEVEL_TABLE="$6"
SAVE_BAM="$7"

# Intermediate output
BAM_OUTPUT="basecalls.bam"

# Step 1: Run dorado basecaller
"$DORADO" basecaller -x "$GPU" --emit-moves \
  --reference "$REFERENCE" \
  "$MODEL" "$READS" > "$BAM_OUTPUT"

# Step 2: Run refine_signals.py
python refine_signals.py \
  --pod5 "$READS" \
  --bam "$BAM_OUTPUT" \
  --level_table "$LEVEL_TABLE" \
  --save_bam "$SAVE_BAM"


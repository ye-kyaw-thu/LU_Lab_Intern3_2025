#!/bin/bash

# Written by Ye Kyaw Thu, LU Lab., Myanmar
# Last Update: 12 June 2025

# Base directory for input files
INPUT_DIR="/home/ye/ye/exp/gpt-mt/nllb/data/squad/train/"

# Directory for output files
OUTPUT_DIR="/home/ye/ye/exp/gpt-mt/nllb/squad-my/"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Iterate over each .src file in the input directory
for FILE in "$INPUT_DIR"/*.txt; do
    # Extract the base filename without the extension
    BASENAME=$(basename "$FILE" .txt)
    
    # Define the output file name
    OUTPUT_FILE="$OUTPUT_DIR/$BASENAME.my"
    
    # Print the command being executed (for debugging)
    echo "Running nllb-translate.sh for $FILE"
    
    # Run the translation command
    time ./nllb-translate.sh --input "$FILE" --source eng_Latn --target mya_Mymr --output "$OUTPUT_FILE"
done


#!/bin/bash

# Written by Ye, LU Lab., Myanmar
# for extracting SQuAD fields  

# Function to process a single CSV file
process_file() {
    local input_file=$1
    local prefix=$2
    
    echo "Processing $input_file..."
    
    # Get all columns from the CSV file
    columns=$(python3 -c "
import csv
with open('$input_file', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    print(' '.join(reader.fieldnames))
")
    
    # Extract each column
    for col in $columns; do
        output_file="${prefix}_${col}.txt"
        python3 extract_csv_field.py --input "$input_file" --column "$col" --output "$output_file"
    done
}

# Main script
echo "Starting SQUAD dataset extraction..."

# Process training file
process_file "train-00000-of-00001.csv" "train"

# Process validation file
process_file "validation-00000-of-00001.csv" "valid"

echo "Extraction completed successfully."


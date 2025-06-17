#!/usr/bin/env python3
import argparse
import csv
import sys
import os

def clean_text(text):
    """Remove newlines and extra spaces from text"""
    if not text:
        return text
    # Replace newlines with spaces
    text = text.replace('\r', ' ').replace('\n', ' ')
    # Collapse multiple spaces into one
    return ' '.join(text.split())

def extract_field(input_file, column, output_file=None):
    try:
        with open(input_file, 'r', encoding='utf-8') as csvfile:
            # Use csv reader to properly handle quoted fields with commas
            reader = csv.DictReader(csvfile)
            
            if column not in reader.fieldnames:
                print(f"Error: Column '{column}' not found in CSV file. Available columns: {', '.join(reader.fieldnames)}")
                sys.exit(1)
                
            data = []
            for row in reader:
                field_value = row[column]
                # Clean the text by removing internal newlines
                cleaned_value = clean_text(field_value)
                data.append(cleaned_value)
            
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as outfile:
                    for item in data:
                        outfile.write(item + '\n')
                print(f"Successfully extracted '{column}' to {output_file}")
            else:
                for item in data:
                    print(item)
                    
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Extract specific fields from SQUAD CSV files')
    parser.add_argument('--input', required=True, help='Input CSV filename')
    parser.add_argument('--column', required=True, help='Column name to extract')
    parser.add_argument('--output', help='Output filename (optional)')
    
    args = parser.parse_args()
    
    extract_field(args.input, args.column, args.output)

if __name__ == '__main__':
    main()


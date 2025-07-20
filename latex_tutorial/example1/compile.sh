#!/bin/bash

# Check if filename is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <filename_without_extension>"
  exit 1
fi

BASENAME="$1"

# First compilation with xelatex
xelatex "$BASENAME.tex"

# Run biber to process bibliography
biber "$BASENAME"

# Second compilation with xelatex
xelatex "$BASENAME.tex"

# Optional third pass to resolve all references
xelatex "$BASENAME.tex"

echo "Compilation complete: $BASENAME.pdf"




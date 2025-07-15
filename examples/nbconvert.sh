#!/bin/bash

# Create the output directory if it does not exist
mkdir -p nbconverted

# Convert Jupyter notebooks to Python scripts in the nbconverted folder
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb
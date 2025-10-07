#!/usr/bin/env bash
# Install Python dependencies for the radar chart script

set -e  # Exit on first error

echo "Installing Python dependencies using pip3..."

# Upgrade pip first
python3 -m pip install --upgrade pip

# Install required libraries
pip3 install \
    pandas \
    numpy \
    matplotlib

echo "All dependencies installed successfully."


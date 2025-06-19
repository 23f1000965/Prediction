#!/bin/bash

echo "Setting up Python 3.10.9 virtual environment..."

# Check if Python 3.10 is installed
if ! command -v python3.10 &> /dev/null; then
    if ! python3 --version | grep -q "Python 3.10"; then
        echo "Python 3.10.x is not installed or not in PATH."
        echo "Please install Python 3.10.9 from https://www.python.org/downloads/release/python-3109/"
        exit 1
    fi
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo "Environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"

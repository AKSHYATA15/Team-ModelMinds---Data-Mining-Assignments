#!/bin/bash

echo "Setting up Python environment..."

python3 -m venv graph_env
source graph_env/bin/activate

# Install core dependencies
pip install numpy
pip install scipy

echo "Environment ready."
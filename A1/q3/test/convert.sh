#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Error: Need 3 arguments"
    echo "Usage: $0 <graphs_file> <patterns_file> <features_file>"
    exit 1
fi

source graph_env/bin/activate

python3 -c "
import sys
sys.path.append('.')
from graph_indexing_optimized import convert_to_features

convert_to_features('$1', '$2', '$3')
"
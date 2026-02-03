#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Error: Need 3 arguments"
    echo "Usage: $0 <db_features> <query_features> <output_file>"
    exit 1
fi

source graph_env/bin/activate

python3 -c "
import sys
sys.path.append('.')
from graph_indexing_optimized import generate_candidates

generate_candidates('$1', '$2', '$3')
"
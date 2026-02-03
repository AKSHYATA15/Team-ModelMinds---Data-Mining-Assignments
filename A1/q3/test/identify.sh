#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 <database_file> <patterns_file> [k]"
    echo "  database_file: path to database graphs"
    echo "  patterns_file: path to output patterns"
    echo "  k: number of patterns to select (default: 50)"
    exit 1
fi

source graph_env/Scripts/activate

K=${3:-50}  # Default to 50 if not provided

python -c "
import sys
sys.path.append('.')
from graph_indexing_optimized import identify_patterns

database_file = '$1'
patterns_file = '$2'
k = $K

identify_patterns(database_file, patterns_file, k=k)
"
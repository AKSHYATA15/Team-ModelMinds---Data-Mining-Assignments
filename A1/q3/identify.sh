#!/bin/bash



if [ $# -ne 2 ]; then
    echo "Usage: $0 <database_file> <patterns_file>"
    echo "  database_file: path to database graphs"
    echo "  patterns_file: path to output patterns"
    exit 1
fi

source graph_env/bin/activate

# Try to find gSpan or other FSM executables
GSPAN_PATH=""
if [ -x "./gspan" ]; then
    GSPAN_PATH="./gspan"
elif [ -x "/usr/local/bin/gspan" ]; then
    GSPAN_PATH="/usr/local/bin/gspan"
fi

python3 -c "
import sys
sys.path.append('.')
from graph_indexing import identify_patterns

database_file = '$1'
patterns_file = '$2'
fsm_path = '$GSPAN_PATH' if '$GSPAN_PATH' else None

identify_patterns(database_file, patterns_file, fsm_path)
"
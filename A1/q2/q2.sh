##llm generated code 

#!/bin/bash

GSPAN_BIN=$1
FSG_BIN=$2
GASTON_BIN=$3
DATASET=$4
OUTDIR=$5

echo "== Checking Python version =="
python3 --version

echo "== Preprocessing dataset =="
rm -rf processed
python3 preprocess_yeast.py "$DATASET" processed

echo "== Running experiments =="
python3 run_experiments.py \
  --gspan "$GSPAN_BIN" \
  --fsg "$FSG_BIN" \
  --gaston "$GASTON_BIN" \
  --dataset "$DATASET" \
  --outdir "$OUTDIR"

echo "== Done =="

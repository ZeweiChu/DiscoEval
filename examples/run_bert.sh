#!/bin/bash

# ./run_bert.sh 0 avg large

hostname
echo "task: "$1
echo "layer: "$2
echo "model type:" $3
python bert.py \
    --task_index $1 \
    --layer $2 \
    --model_type $3

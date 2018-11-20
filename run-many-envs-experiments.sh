#!/bin/bash

ARGS=('HandEye2-v0 100 20 10' 'HandEye3-v0 100 50 10' 'HandEye4-v0 50 150 10' 'HandEye5-v0 10 300 10')
num_el=4
source venv/bin/activate
cd docs/source/notebooks/handeye
for ((i=0; i < $num_el; i++)); do
    python mean_results_handeye.py ${ARGS[i]}
done

#!/bin/bash

MAZES=('MazeF3')
num_el=6
source venv/bin/activate
cd examples/acs2/maze
for ((i=0; i < $num_el; i++)); do
	python mean_results_maze.py ${MAZES[i]} 10 400 10
done

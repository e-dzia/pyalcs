#!/bin/bash

MAZES=('Woods14-v0' 'MazeF4-v0' 'MazeF1-v0' 'MazeF2-v0' 'MazeF3-v0' 'Woods1-v0')
num_el=6
source venv/bin/activate
cd examples/acs2/maze
for ((i=0; i < $num_el; i++)); do
	python mean_results_maze.py ${MAZES[i]} 10 400 10
done

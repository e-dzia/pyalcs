#!/bin/bash

MAZES=('Maze6-v0' 'MazeF4-v0')
num_el=2
source venv/bin/activate
cd examples/acs2/maze
for ((i=0; i < $num_el; i++)); do
	python mean_results_maze.py ${MAZES[i]} 10 800 10
done

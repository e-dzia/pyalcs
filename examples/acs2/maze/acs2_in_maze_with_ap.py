import logging

import gym
import sys

from examples.acs2.maze.acs2_in_maze import maze_metrics

sys.path.insert(0, '/home/e-dzia/openai-envs')
# noinspection PyUnresolvedReferences
import gym_maze

from lcs.agents.acs2 import ACS2, Configuration

# Configure logger
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("integration")


if __name__ == '__main__':

    # Load desired environment
    maze = gym.make('Maze4-v0')

    # Configure and create the agent
    cfg = Configuration(maze.observation_space.n, maze.action_space.n,
                        epsilon=1.0,
                        do_ga=False,
                        do_action_planning=True,
                        action_planning_frequency=30,
                        user_metrics_collector_fcn=maze_metrics)
    logger.info(cfg)

    # Explore the environment
    agent = ACS2(cfg)
    population, explore_metrics = agent.explore(maze, 500)

    # Exploit the environment
    agent = ACS2(cfg, population)
    population, exploit_metric = agent.exploit(maze, 10)

    for metric in exploit_metric:
        logger.info(metric)

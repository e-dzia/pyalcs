# Plot constants
import argparse
import datetime
import os

import gym
import sys

from examples.acs2.maze.acs2_in_maze import maze_metrics

sys.path.insert(0, '/home/e-dzia/openai-envs')
# noinspection PyUnresolvedReferences
import gym_maze

from lcs.agents.acs2 import ACS2, ClassifiersList, Configuration
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys

TITLE_TEXT_SIZE = 24
AXIS_TEXT_SIZE = 18
LEGEND_TEXT_SIZE = 16


def parse_metrics_to_df(explore_metrics, exploit_metrics):
    def extract_details(row):
        row['trial'] = row['trial']
        row['steps'] = row['steps_in_trial']
        row['numerosity'] = row['population']['numerosity']
        row['reliable'] = row['population']['reliable']
        row['knowledge'] = row['knowledge']
        return row

    # Load both metrics into data frame
    explore_df = pd.DataFrame(explore_metrics)
    exploit_df = pd.DataFrame(exploit_metrics)

    # Mark them with specific phase
    explore_df['phase'] = 'explore'
    exploit_df['phase'] = 'exploit'

    # Extract details
    explore_df = explore_df.apply(extract_details, axis=1)
    exploit_df = exploit_df.apply(extract_details, axis=1)

    # Adjuts exploit trial counter
    exploit_df['trial'] = exploit_df.apply(
        lambda r: r['trial'] + len(explore_df), axis=1)

    # Concatenate both dataframes
    df = pd.concat([explore_df, exploit_df])
    df.drop(['population'], axis=1, inplace=True)
    df.set_index('trial', inplace=True)

    return df


def plot_knowledge(df, ax=None, additional_info=""):
    if ax is None:
        ax = plt.gca()

    explore_df = df.query("phase == 'explore'")
    exploit_df = df.query("phase == 'exploit'")

    explore_df['knowledge'].plot(ax=ax, c='blue')
    exploit_df['knowledge'].plot(ax=ax, c='red')
    ax.axvline(x=len(explore_df), c='black', linestyle='dashed')

    ax.set_title("Achieved knowledge{}".format(additional_info),
                 fontsize=TITLE_TEXT_SIZE)
    ax.set_xlabel("Trial", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylabel("Knowledge [%]", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylim([0, 105])
    ax.legend(fontsize=LEGEND_TEXT_SIZE)


def plot_classifiers(df, ax=None):
    if ax is None:
        ax = plt.gca()

    explore_df = df.query("phase == 'explore'")
    exploit_df = df.query("phase == 'exploit'")

    df['numerosity'].plot(ax=ax, c='blue')
    df['reliable'].plot(ax=ax, c='red')

    ax.axvline(x=len(explore_df), c='black', linestyle='dashed')

    ax.set_title("Classifiers", fontsize=TITLE_TEXT_SIZE)
    ax.set_xlabel("Trial", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylabel("Classifiers", fontsize=AXIS_TEXT_SIZE)
    ax.legend(fontsize=LEGEND_TEXT_SIZE)


def plot_performance(metrics_df, env_name, additional_info,
                     with_ap=""):
    plt.figure(figsize=(13, 10), dpi=100)
    plt.suptitle(f'ACS2 Performance in {env_name} environment '
                 f'{additional_info}', fontsize=32)

    ax2 = plt.subplot(211)
    plot_knowledge(metrics_df, ax2, with_ap)

    ax3 = plt.subplot(212)
    plot_classifiers(metrics_df, ax3)

    plt.subplots_adjust(top=0.86, wspace=0.3, hspace=0.3)


def plot_both_performances(metrics_ap, metrics_no_ap, env_name,
                           additional_info):
    plt.figure(figsize=(13, 10), dpi=100)
    plt.suptitle(f'ACS2 Performance in {env_name} environment '
                 f'{additional_info}', fontsize=32)

    ax2 = plt.subplot(211)
    plot_knowledge(metrics_ap, ax2, ", with Action Planning")

    ax3 = plt.subplot(212)
    plot_knowledge(metrics_no_ap, ax3, ", without Action Planning")

    plt.subplots_adjust(top=0.86, wspace=0.3, hspace=0.3)


def mean(i, row_mean, row, first, second=None):
    if second is not None:
        return (row_mean[first][second] * i + row[first][second]) / (i + 1)
    else:
        return (row_mean[first] * i + row[first]) / (i + 1)


def count_mean_values(i: int, metrics, mean_metrics):
    new_metrics = metrics.copy()
    for row, row_new, row_mean in zip(metrics, new_metrics, mean_metrics):
        row_new['knowledge'] = mean(i, row_mean, row, 'knowledge')
        row_new['population']['numerosity'] = mean(i, row_mean, row,
                                                   'population', 'numerosity')
        row_new['steps_in_trial'] = mean(i, row_mean, row, 'steps_in_trial')
        row_new['population']['reliable'] = mean(i, row_mean, row,
                                                 'population', 'reliable')
    return new_metrics


def plot_handeye_mean(number_of_tests=50, env_name='Maze4-v0',
                      filename='mean_results/Maze4-v0.pdf',
                      do_action_planning=True,
                      number_of_trials_explore=30,
                      number_of_trials_exploit=10, theta_r=90):
    hand_eye = gym.make(env_name)
    cfg = Configuration(hand_eye.observation_space.n, hand_eye.action_space.n,
                        epsilon=1.0,
                        theta_r=theta_r,
                        do_ga=False,
                        do_action_planning=do_action_planning,
                        action_planning_frequency=30,
                        metrics_trial_frequency=1,
                        user_metrics_collector_fcn=maze_metrics)

    mean_metrics_he_exploit = []
    mean_metrics_he_explore = []

    for i in range(number_of_tests):
        # the below line can be un-commented for experiments
        print(i, datetime.datetime.now())

        # explore
        agent_he = ACS2(cfg)
        population_he_explore, metrics_he_explore = agent_he.explore(
            hand_eye, number_of_trials_explore)

        # exploit
        agent_he = ACS2(cfg, population_he_explore)
        _, metrics_he_exploit = agent_he.exploit(hand_eye,
                                                 number_of_trials_exploit)

        mean_metrics_he_explore = count_mean_values(i, metrics_he_explore,
                                                    mean_metrics_he_explore)
        mean_metrics_he_exploit = count_mean_values(i, metrics_he_exploit,
                                                    mean_metrics_he_exploit)

    he_metrics_df = parse_metrics_to_df(mean_metrics_he_explore,
                                        mean_metrics_he_exploit)
    # if do_action_planning:
    #     with_ap = ", with Action Planning"
    # else:
    #     with_ap = ", without Action Planning"

    # plot_performance(he_metrics_df, env_name,
    #                  '\nmean for {} experiments'.format(number_of_tests),
    #                  with_ap)
    # plt.savefig(filename.replace(" ", "_").replace(':', '.'),
    #             format='pdf', dpi=100)
    return he_metrics_df


def plot_with_without_ap(filename, metrics_ap, metrics_no_ap):
    plot_both_performances(metrics_ap, metrics_no_ap, env_name,
                           '\nmean for {} experiments'.format(number_of_tests))
    plt.savefig(filename.replace(" ", "_").replace(':', '.'),
                format='pdf', dpi=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", default="Maze4-v0")
    parser.add_argument("-n", "--num-tests", default=10, type=int)
    parser.add_argument("--explore-trials", default=400, type=int)
    parser.add_argument("--exploit-trials", default=10, type=int)
    parser.add_argument("--theta-r", default=0.90, type=float)
    parser.add_argument("--tests-type", default=2, type=int) # 0 - ap,
                                                        # 1 - no ap, 2 - both
    args = parser.parse_args()

    env_name = args.environment
    number_of_tests = args.num_tests
    number_of_trials_explore = args.explore_trials
    number_of_trials_exploit = args.exploit_trials
    theta_r = args.theta_r
    tests_type = args.tests_type

    if tests_type > 2 or tests_type < 0:
        tests_type = 2

    print("Env: {}, Experiments: {}, Explore: {}, Exploit: {},"
          "Theta_r: {}, Tests: {}".format(
          env_name, number_of_tests, number_of_trials_explore,
          number_of_trials_exploit, theta_r, tests_type))

    if not os.path.exists("mean_results"):
        os.makedirs("mean_results")

    start = datetime.datetime.now()
    print("time start: {}".format(start))

    if tests_type == 0 or tests_type == 2:
        metrics_ap = plot_handeye_mean(
            number_of_tests, env_name, do_action_planning=True,
            number_of_trials_explore=number_of_trials_explore,
            number_of_trials_exploit=number_of_trials_exploit,
            theta_r=theta_r)

        metrics_ap.to_csv('mean_results/{}_{}_ap_{}_{}.csv'.
                          format(theta_r, env_name, number_of_tests, start).
                          replace(' ', '_').replace(':', '.'))

    middle = datetime.datetime.now()
    print("done with AP, time: {}, elapsed: {}".format(middle, middle - start))

    if tests_type == 1 or tests_type == 2:
        metrics_no_ap = plot_handeye_mean(
            number_of_tests, env_name, do_action_planning=False,
            number_of_trials_explore=number_of_trials_explore,
            number_of_trials_exploit=number_of_trials_exploit,
            theta_r=theta_r)

        metrics_no_ap.to_csv('mean_results/{}_{}_no_ap_{}_{}.csv'.
                             format(theta_r, env_name, number_of_tests, start).
                             replace(' ', '_').replace(':', '.'))

    end = datetime.datetime.now()
    print("done without AP, time: {}, elapsed: {}".format(end, end - middle))

    # plot_with_without_ap('mean_results/b30_{}_both_{}_{}.pdf'.format(
    #    env_name, number_of_tests, start).replace(' ', '_').replace(':', '.'),
    #                     metrics_ap, metrics_no_ap)

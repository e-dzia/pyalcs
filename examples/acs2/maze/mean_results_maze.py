# Plot constants
import datetime
import os

import gym
import sys
sys.path.insert(0, '/home/e-dzia/openai-envs')
# noinspection PyUnresolvedReferences
import gym_maze

from lcs.agents.acs2 import ACS2, ClassifiersList, Configuration
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys

from examples.acs2.maze.utils import calculate_performance

TITLE_TEXT_SIZE = 24
AXIS_TEXT_SIZE = 18
LEGEND_TEXT_SIZE = 16


def parse_metrics_to_df(explore_metrics, exploit_metrics):
    def extract_details(row):
        row['trial'] = row['agent']['trial']
        row['steps'] = row['agent']['steps']
        row['numerosity'] = row['agent']['numerosity']
        row['reliable'] = row['agent']['reliable']
        row['knowledge'] = row['performance']['knowledge']
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
    df.drop(['agent', 'environment', 'performance'], axis=1, inplace=True)
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


def mean(i, row_mean, row, first, second):
    return (row_mean[first][second] * i + row[first][second]) / (i + 1)


def count_mean_values(i: int, metrics, mean_metrics):
    new_metrics = metrics.copy()
    for row, row_new, row_mean in zip(metrics, new_metrics, mean_metrics):
        row_new['performance']['knowledge'] = mean(i, row_mean, row,
                                                   'performance', 'knowledge')
        row_new['agent']['numerosity'] = mean(i, row_mean, row,
                                              'agent', 'numerosity')
        row_new['agent']['steps'] = mean(i, row_mean, row,
                                         'agent', 'steps')
        row_new['agent']['reliable'] = mean(i, row_mean, row,
                                            'agent', 'reliable')
    return new_metrics


def plot_handeye_mean(number_of_tests=50, env_name='BMaze4-v0',
                      filename='mean_results/BMaze4-v0.pdf',
                      do_action_planning=True,
                      number_of_trials_explore=30,
                      number_of_trials_exploit=10):
    hand_eye = gym.make(env_name)
    cfg = Configuration(hand_eye.observation_space.n, hand_eye.action_space.n,
                        epsilon=1.0,
                        theta_r=75,
                        do_ga=False,
                        do_action_planning=do_action_planning,
                        action_planning_frequency=30,
                        performance_fcn=calculate_performance)

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
    if do_action_planning:
        with_ap = ", with Action Planning"
    else:
        with_ap = ", without Action Planning"

    plot_performance(he_metrics_df, env_name,
                     '\nmean for {} experiments'.format(number_of_tests),
                     with_ap)
    plt.savefig(filename.replace(" ", "_").replace(':', '.'),
                format='pdf', dpi=100)
    return he_metrics_df


def plot_with_without_ap(filename, metrics_ap, metrics_no_ap):
    plot_both_performances(metrics_ap, metrics_no_ap, env_name,
                           '\nmean for {} experiments'.format(number_of_tests))
    plt.savefig(filename.replace(" ", "_").replace(':', '.'),
                format='pdf', dpi=100)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Not enough args provided, using the defaults.")
        env_name = 'Maze4-v0'
        number_of_tests = 10
        number_of_trials_explore = 400
        number_of_trials_exploit = 10
    else:
        env_name = sys.argv[1]
        number_of_tests = int(sys.argv[2])
        number_of_trials_explore = int(sys.argv[3])
        number_of_trials_exploit = int(sys.argv[4])

    print("Env: {}, Experiments: {}, Explore: {}, Exploit: {}".format(
          env_name, number_of_tests, number_of_trials_explore,
          number_of_trials_exploit))

    #os.chdir("/".join(sys.argv[0].split("/")[:-1]))

    start = datetime.datetime.now()
    print("time start: {}".format(start))

    metrics_ap = plot_handeye_mean(
        number_of_tests, env_name, 'mean_results/b30_{}_ap_{}_{}.pdf'
        .format(env_name, number_of_tests, start).replace(' ', '_').
        replace(':', '.'), do_action_planning=True,
        number_of_trials_explore=number_of_trials_explore,
        number_of_trials_exploit=number_of_trials_exploit)

    middle = datetime.datetime.now()
    print("done with AP, time: {}, elapsed: {}".format(middle, middle - start))

    metrics_ap.to_csv('mean_results/b30_{}_ap_{}_{}.csv'.
                      format(env_name, number_of_tests, start).
                      replace(' ', '_').replace(':', '.'))

    metrics_no_ap = plot_handeye_mean(
        number_of_tests, env_name, 'mean_results/b30_{}_no_ap_{}_{}.pdf'.
        format(env_name, number_of_tests, start).replace(' ', '_').
        replace(':', '.'), do_action_planning=False,
        number_of_trials_explore=number_of_trials_explore,
        number_of_trials_exploit=number_of_trials_exploit)

    end = datetime.datetime.now()
    print("done without AP, time: {}, elapsed: {}".format(end, end - middle))

    metrics_no_ap.to_csv('mean_results/b30_{}_no_ap_{}_{}.csv'.
                         format(env_name, number_of_tests, start).
                         replace(' ', '_').replace(':', '.'))

    # plot_with_without_ap('mean_results/b30_{}_both_{}_{}.pdf'.format(
    #    env_name, number_of_tests, start).replace(' ', '_').replace(':', '.'),
    #                     metrics_ap, metrics_no_ap)

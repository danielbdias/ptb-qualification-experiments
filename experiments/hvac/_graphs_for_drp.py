

from _utils import load_data, load_time_csv

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(title, no_heuristic_stats, heuristic_stats):
    no_heuristic_iterations = list(map(lambda item : item.iteration, no_heuristic_stats))
    no_heuristic_best_returns = list(map(lambda item : float(item.best_return) * -1, no_heuristic_stats))
    no_heuristic_train_returns = list(map(lambda item : float(item.train_return) * -1, no_heuristic_stats))

    heuristic_iterations = list(map(lambda item : item.iteration, heuristic_stats))
    heuristic_best_returns = list(map(lambda item : float(item.best_return) * -1, heuristic_stats))
    heuristic_train_returns = list(map(lambda item : float(item.train_return) * -1, heuristic_stats))
    
    plt.subplots(1, figsize=(10,10))
    plt.plot(no_heuristic_iterations, no_heuristic_best_returns, '--', label="No Heuristic (Best)")
    plt.plot(no_heuristic_iterations, no_heuristic_train_returns, '--', label="No Heuristic (Train)")
    plt.plot(heuristic_iterations, heuristic_best_returns, label="Heuristic (Best)")
    plt.plot(heuristic_iterations, heuristic_train_returns, label="Heuristic (Train)")

    plt.title(title)
    plt.xlabel("Iterations"), plt.ylabel("Costs"), plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig(f'{root_folder}/zzz_graph_learningcurve_deepreactive.png', format='png')

def plot_time_bars(title, time_data):
    bar_names = (
        "No Heuristic",
        "Heuristic",
    )
    time_sizes = {
        "Deterministic": np.array([ 0, time_data["Deterministic - Straight line"] ]),
        "Probabilistic": np.array([ time_data["Probabilistic (no heuristic) - Straight line"], time_data["Probabilistic + Heuristic - Straight line"] ]),
    }
    width = 0.5

    plt.subplots(1, figsize=(10,10))
    bottom = np.zeros(2)

    for exp_type, total_time in time_sizes.items():
        plt.bar(bar_names, total_time, width, label=exp_type, bottom=bottom)
        bottom += total_time

    plt.title(title)
    plt.ylabel("Time"), plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig(f'{root_folder}/zzz_graph_time_deepreactive.png', format='png')

print('--------------------------------------------------------------------------------')
print('Generating graphs for DRP')
print('--------------------------------------------------------------------------------')
print()

root_folder = os.path.dirname(__file__)
no_heuristic_stats = load_data(f'{root_folder}/zzz_probabilistic_no_heuristic_deepreactive_statistics.pickle')
heuristic_stats = load_data(f'{root_folder}/zzz_probabilistic_with_heuristic_deepreactive_statistics.pickle')

plot_learning_curve("Learning Curve - DRP", no_heuristic_stats, heuristic_stats)

time_data = load_time_csv(f'{root_folder}/zzz_deepreactive_time.csv')

plot_time_bars("Experiment time - DRP", time_data)
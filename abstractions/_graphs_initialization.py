from _utils import load_data
from _graphs import stat_curves

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_cost_curve_per_iteration(title, planner, regular_stats, heuristic_stats, cost_getter):
    regular_iterations, regular_best_return_curves_mean, regular_best_return_curves_stddev = stat_curves(regular_stats, cost_getter)
    heuristic_iterations, heuristic_best_return_curves_mean, heuristic_best_return_curves_stddev = stat_curves(heuristic_stats, cost_getter)
    
    plt.subplots(1, figsize=(8,5))
    plt.plot(regular_iterations, regular_best_return_curves_mean, '--', label=f'PtB (regular)')
    plt.fill_between(regular_iterations, (regular_best_return_curves_mean - regular_best_return_curves_stddev), (regular_best_return_curves_mean + regular_best_return_curves_stddev), alpha=0.2)

    plt.plot(heuristic_iterations, heuristic_best_return_curves_mean, label=f'PtB (heuristic)')
    plt.fill_between(heuristic_iterations, (heuristic_best_return_curves_mean - heuristic_best_return_curves_stddev), (heuristic_best_return_curves_mean + heuristic_best_return_curves_stddev), alpha=0.2)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Costs", fontsize=14)
    plt.legend(loc="best", fontsize=14)
    plt.tight_layout()

    plt.rc('font', family='serif')

    plt.savefig(f'{root_folder}/_plots/{domain_name}_{planner}.pdf', format='pdf')

def plot_time_bars(title, planner, regular_stats, heuristic_stats, simplified_stats):
    regular_mean_elapsed_time = np.mean(list(map(lambda item : item.elapsed_time, regular_stats)))
    heuristic_mean_elapsed_time = np.mean(list(map(lambda item : item.elapsed_time, heuristic_stats)))
    simplified_mean_elapsed_time = np.mean(list(map(lambda item : item.elapsed_time, simplified_stats)))

    bar_names = (
        "PtB",
        "PtB-Heuristic",
    )
    time_sizes = {
        "Relaxed": np.array([ 0, simplified_mean_elapsed_time ]),
        "Original": np.array([ regular_mean_elapsed_time, heuristic_mean_elapsed_time ]),
    }
    width = 0.7

    plt.subplots(1, figsize=(7,8))
    bottom = np.zeros(2)

    for exp_type, total_time in time_sizes.items():
        plt.bar(bar_names, total_time, width, label=exp_type, bottom=bottom)
        bottom += total_time

    plt.title(title, fontsize=22, fontweight='bold')
    plt.ylabel("Time", fontsize=18)
    plt.legend(loc="best", fontsize=20)
    plt.tight_layout()

    #plt.rc('xtick', labelsize=30)
    plt.rcParams.update({'font.size':15})
    plt.rc('font', family='serif')

    plt.savefig(f'{root_folder}/_plots/{domain_name}_graph_time_{planner}.pdf', format='pdf')

print('--------------------------------------------------------------------------------')
print('Generating graphs')
print('--------------------------------------------------------------------------------')
print()

root_folder = os.path.dirname(__file__)
# domain_name = sys.argv[1]
domain_name = 'UAV'

regular_straightline_stats = load_data(f'{root_folder}/_results/{domain_name}_regular_statistics.pickle')
simplified_straightline_stats = load_data(f'{root_folder}/_results/{domain_name}_simplified_statistics.pickle')
heuristic_straightline_stats = load_data(f'{root_folder}/_results/{domain_name}_heuristic_statistics.pickle')

plot_cost_curve_per_iteration(f'Best Costs per Iteration ({domain_name})', 'slp', regular_straightline_stats, heuristic_straightline_stats, lambda item : -item.best_return, cost_curve_per_iteration_path)
plot_time_bars(f'Mean execution time ({domain_name})', 'straightline', regular_straightline_stats, heuristic_straightline_stats, simplified_straightline_stats)

print('done!')
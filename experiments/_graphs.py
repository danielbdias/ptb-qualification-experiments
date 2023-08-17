from _utils import load_data, load_time_csv

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def get_curves(experiment_summaries, attribute_getter):
    curves = []

    for experiment_summary in experiment_summaries:
        curve = np.array(list(map(attribute_getter, experiment_summary.statistics_history)))
        curves.append(curve)

    return curves

def stat_curves(experiment_summaries, attribute_getter):
    curves = get_curves(experiment_summaries, attribute_getter)

    # convert "list of np.array" to "np.array of np.array"
    curves = np.stack(curves)

    curves_mean = np.mean(curves, axis=0)
    curves_stddev = np.std(curves, axis=0)

    return curves_mean, curves_stddev
    

def plot_cost_curve_per_iteration(title, metric_label, metric_name, planner, no_heuristic_stats, heuristic_stats, cost_getter):
    iterations_curves = get_curves(no_heuristic_stats, lambda item : item.iteration)
    iterations = iterations_curves[0] # all curves are the same
    
    no_heuristic_best_return_curves_mean, no_heuristic_best_return_curves_stddev = stat_curves(no_heuristic_stats, cost_getter)
    heuristic_best_return_curves_mean, heuristic_best_return_curves_stddev = stat_curves(heuristic_stats, cost_getter)
    
    plt.subplots(1, figsize=(10,10))
    plt.plot(iterations, no_heuristic_best_return_curves_mean, '--', label=f'No Heuristic ({metric_label})')
    plt.fill_between(iterations, (no_heuristic_best_return_curves_mean - no_heuristic_best_return_curves_stddev), (no_heuristic_best_return_curves_mean + no_heuristic_best_return_curves_stddev), alpha=0.2)

    plt.plot(iterations, heuristic_best_return_curves_mean, label=f'Heuristic ({metric_label})')
    plt.fill_between(iterations, (heuristic_best_return_curves_mean - heuristic_best_return_curves_stddev), (heuristic_best_return_curves_mean + heuristic_best_return_curves_stddev), alpha=0.2)

    plt.title(title)
    plt.xlabel("Iterations"), plt.ylabel("Costs"), plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig(f'{root_folder}/_plots/{domain_name}_{metric_name}_{planner}.png', format='png')

def plot_time_bars(title, planner, no_heuristic_stats, heuristic_probabilistic_stats, heuristic_deterministic_stats):
    no_heuristic_mean_elapsed_time = np.mean(list(map(lambda item : item.elapsed_time, no_heuristic_stats)))
    heuristic_probabilistic_mean_elapsed_time = np.mean(list(map(lambda item : item.elapsed_time, heuristic_probabilistic_stats)))
    heuristic_deterministic_mean_elapsed_time = np.mean(list(map(lambda item : item.elapsed_time, heuristic_deterministic_stats)))

    bar_names = (
        "No Heuristic",
        "Heuristic",
    )
    time_sizes = {
        "Deterministic": np.array([ 0, heuristic_deterministic_mean_elapsed_time ]),
        "Probabilistic": np.array([ no_heuristic_mean_elapsed_time, heuristic_probabilistic_mean_elapsed_time ]),
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

    plt.savefig(f'{root_folder}/_plots/{domain_name}_graph_time_{planner}.png', format='png')

print('--------------------------------------------------------------------------------')
print('Generating graphs')
print('--------------------------------------------------------------------------------')
print()

root_folder = os.path.dirname(__file__)
domain_name = sys.argv[1]

# no_heuristic_drp_stats = load_data(f'{root_folder}/{domain_name}_no_heuristic_deepreactive_probabilistic_statistics.pickle')
# heuristic_deterministic_drp_stats = load_data(f'{root_folder}/{domain_name}_no_heuristic_deepreactive_deterministic_statistics.pickle')
# heuristic_probabilistic_drp_stats = load_data(f'{root_folder}/{domain_name}_no_heuristic_deepreactive_probabilistic_statistics.pickle')

no_heuristic_straightline_stats = load_data(f'{root_folder}/_results/{domain_name}_no_heuristic_straightline_probabilistic_statistics.pickle')
heuristic_probabilistic_straightline_stats = load_data(f'{root_folder}/_results/{domain_name}_heuristic_straightline_probabilistic_statistics.pickle')
heuristic_deterministic_straightline_stats = load_data(f'{root_folder}/_results/{domain_name}_heuristic_straightline_deterministic_statistics.pickle')

plot_cost_curve_per_iteration(f'Best Costs per Iteration - Straightline ({domain_name})', 'Best', 'best_returns', 'straightline', no_heuristic_straightline_stats, heuristic_probabilistic_straightline_stats, lambda item : -item.best_return)
plot_cost_curve_per_iteration(f'Train Costs per Iteration - Straightline ({domain_name})', 'Train', 'train_returns', 'straightline', no_heuristic_straightline_stats, heuristic_probabilistic_straightline_stats, lambda item : -item.train_return)
plot_time_bars(f'Experiment time - Straightline ({domain_name})', 'straightline', no_heuristic_straightline_stats, heuristic_probabilistic_straightline_stats, heuristic_deterministic_straightline_stats)

if os.path.exists(f'{root_folder}/{domain_name}_no_heuristic_deepreactive_probabilistic_statistics.pickle'):
    no_heuristic_drp_stats = load_data(f'{root_folder}/{domain_name}_no_heuristic_deepreactive_probabilistic_statistics.pickle')
    heuristic_deterministic_drp_stats = load_data(f'{root_folder}/{domain_name}_no_heuristic_deepreactive_deterministic_statistics.pickle')
    heuristic_probabilistic_drp_stats = load_data(f'{root_folder}/{domain_name}_no_heuristic_deepreactive_probabilistic_statistics.pickle')

    plot_cost_curve_per_iteration(f'Best Costs per Iteration - DRP ({domain_name})', 'Best', 'best_returns', 'drp', no_heuristic_straightline_stats, heuristic_probabilistic_straightline_stats, lambda item : -item.best_return)
    plot_cost_curve_per_iteration(f'Train Costs per Iteration - DRP ({domain_name})', 'Train', 'train_returns', 'drp', no_heuristic_straightline_stats, heuristic_probabilistic_straightline_stats, lambda item : -item.train_return)
    plot_time_bars(f'Experiment time - DRP ({domain_name})', 'drp', no_heuristic_straightline_stats, heuristic_probabilistic_straightline_stats, heuristic_deterministic_straightline_stats)

print('done!')
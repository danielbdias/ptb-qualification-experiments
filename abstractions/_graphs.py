from _utils import load_data

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
    iteration_curves = get_curves(experiment_summaries, lambda item : item.iteration)
    curves = get_curves(experiment_summaries, attribute_getter)

    iteration_curve_max_len = -1
    larger_iteration_curve = None

    # find experiment with more iterations
    for i in range(len(experiment_summaries)):
        iteration_curve = iteration_curves[i]

        if len(iteration_curve) > iteration_curve_max_len:
            iteration_curve_max_len = len(iteration_curve)
            larger_iteration_curve = iteration_curve

    # repeat last value for each curve with less iterations
    resized_curves = []

    for i in range(len(experiment_summaries)):
        curve = curves[i]
        size_diff = iteration_curve_max_len - len(curve)
        if size_diff <= 0:
            resized_curves.append(curve)
        else:
            curve_last_value = curve[-1]
            resized_curve = np.append(curve, np.repeat(curve_last_value, size_diff))
            resized_curves.append(resized_curve)

    # convert "list of np.array" to "np.array of np.array"
    resized_curves = np.stack(resized_curves)

    curves_mean = np.mean(resized_curves, axis=0)
    curves_stddev = np.std(resized_curves, axis=0)

    return larger_iteration_curve, curves_mean, curves_stddev
    

def plot_cost_curve_per_iteration(title, metric_name, planner, regular_stats, simplified_stats, cost_getter):
    regular_iterations, regular_best_return_curves_mean, regular_best_return_curves_stddev = stat_curves(regular_stats, cost_getter)
    simplified_iterations, simplified_best_return_curves_mean, simplified_best_return_curves_stddev = stat_curves(simplified_stats, cost_getter)
    
    plt.subplots(1, figsize=(8,5))
    plt.plot(regular_iterations, regular_best_return_curves_mean, '--', label=f'PtB (regular)')
    plt.fill_between(regular_iterations, (regular_best_return_curves_mean - regular_best_return_curves_stddev), (regular_best_return_curves_mean + regular_best_return_curves_stddev), alpha=0.2)

    plt.plot(simplified_iterations, simplified_best_return_curves_mean, label=f'PtB (simplified)')
    plt.fill_between(simplified_iterations, (simplified_best_return_curves_mean - simplified_best_return_curves_stddev), (simplified_best_return_curves_mean + simplified_best_return_curves_stddev), alpha=0.2)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Costs", fontsize=14)
    plt.legend(loc="best", fontsize=14)
    plt.tight_layout()

    plt.rc('font', family='serif')

    plt.savefig(f'{root_folder}/_plots/{domain_name}_{metric_name}_{planner}.pdf', format='pdf')

print('--------------------------------------------------------------------------------')
print('Generating graphs')
print('--------------------------------------------------------------------------------')
print()

root_folder = os.path.dirname(__file__)
# domain_name = sys.argv[1]
domain_name = 'UAV'

regular_straightline_stats = load_data(f'{root_folder}/_results/{domain_name}_regular_statistics.pickle')
simplified_straightline_stats = load_data(f'{root_folder}/_results/{domain_name}_simplified_statistics.pickle')

plot_cost_curve_per_iteration(f'Best Costs per Iteration ({domain_name})', 'best_returns', 'straightline', regular_straightline_stats, simplified_straightline_stats, lambda item : -item.best_return)

print('done!')
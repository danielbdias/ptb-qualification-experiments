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
    

def plot_cost_curve_per_iteration(title, metric_label, metric_name, planner, no_heuristic_stats, heuristic_stats, mixed_heuristic_stats, cost_getter):
    no_heuristic_iterations, no_heuristic_best_return_curves_mean, no_heuristic_best_return_curves_stddev = stat_curves(no_heuristic_stats, cost_getter)
    heuristic_iterations, heuristic_best_return_curves_mean, heuristic_best_return_curves_stddev = stat_curves(heuristic_stats, cost_getter)
    mixed_heuristic_iterations, mixed_heuristic_best_return_curves_mean, mixed_heuristic_best_return_curves_stddev = stat_curves(mixed_heuristic_stats, cost_getter)
    
    plt.subplots(1, figsize=(8,5))
    plt.plot(no_heuristic_iterations, no_heuristic_best_return_curves_mean, '--', label=f'PtB-Stochastic')
    plt.fill_between(no_heuristic_iterations, (no_heuristic_best_return_curves_mean - no_heuristic_best_return_curves_stddev), (no_heuristic_best_return_curves_mean + no_heuristic_best_return_curves_stddev), alpha=0.2)

    plt.plot(heuristic_iterations, heuristic_best_return_curves_mean, label=f'PtB-Heuristic (mean)')
    plt.fill_between(heuristic_iterations, (heuristic_best_return_curves_mean - heuristic_best_return_curves_stddev), (heuristic_best_return_curves_mean + heuristic_best_return_curves_stddev), alpha=0.2)

    plt.plot(mixed_heuristic_iterations, mixed_heuristic_best_return_curves_mean, label=f'PtB-Heuristic (combined)')
    plt.fill_between(mixed_heuristic_iterations, (mixed_heuristic_best_return_curves_mean - mixed_heuristic_best_return_curves_stddev), (mixed_heuristic_best_return_curves_mean + mixed_heuristic_best_return_curves_stddev), alpha=0.2)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Costs", fontsize=14)
    plt.legend(loc="best", fontsize=14)
    plt.tight_layout()

    plt.rc('font', family='serif')

    plt.savefig(f'{root_folder}/_plots/{domain_name}_{metric_name}_{planner}.pdf', format='pdf')

def plot_time_bars(title, planner, no_heuristic_stats, heuristic_probabilistic_stats, heuristic_deterministic_stats):
    no_heuristic_mean_elapsed_time = np.mean(list(map(lambda item : item.elapsed_time, no_heuristic_stats)))
    heuristic_probabilistic_mean_elapsed_time = np.mean(list(map(lambda item : item.elapsed_time, heuristic_probabilistic_stats)))
    heuristic_deterministic_mean_elapsed_time = np.mean(list(map(lambda item : item.elapsed_time, heuristic_deterministic_stats)))

    bar_names = (
        "PtB",
        "PtB-Heuristic",
    )
    time_sizes = {
        "Relaxed": np.array([ 0, heuristic_deterministic_mean_elapsed_time ]),
        "Original": np.array([ no_heuristic_mean_elapsed_time, heuristic_probabilistic_mean_elapsed_time ]),
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
domain_name = sys.argv[1]

no_heuristic_straightline_stats = load_data(f'{root_folder}/_results/{domain_name}_no_heuristic_straightline_probabilistic_statistics.pickle')
heuristic_probabilistic_straightline_stats = load_data(f'{root_folder}/_results/{domain_name}_heuristic_straightline_probabilistic_statistics.pickle')
heuristic_deterministic_straightline_stats = load_data(f'{root_folder}/_results/{domain_name}_heuristic_straightline_deterministic_statistics.pickle')

mixed_heuristic_probabilistic_straightline_stats = load_data(f'{root_folder}/_results/{domain_name}_mixed_heuristic_straightline_probabilistic_statistics.pickle')
mixed_heuristic_deterministic_mean_straightline_stats = load_data(f'{root_folder}/_results/{domain_name}_mixed_heuristic_straightline_mean_deterministic_statistics.pickle')
mixed_heuristic_deterministic_max_straightline_stats = load_data(f'{root_folder}/_results/{domain_name}_mixed_heuristic_straightline_max_deterministic_statistics.pickle')

plot_cost_curve_per_iteration(f'Best Costs per Iteration ({domain_name})', 'Best', 'best_returns', 'straightline', no_heuristic_straightline_stats, heuristic_probabilistic_straightline_stats, mixed_heuristic_probabilistic_straightline_stats, lambda item : -item.best_return)
# plot_cost_curve_per_iteration(f'Train Costs per Iteration ({domain_name})', 'Train', 'train_returns', 'straightline', no_heuristic_straightline_stats, heuristic_probabilistic_straightline_stats, lambda item : -item.train_return)
# plot_time_bars(f'Mean execution time ({domain_name})', 'straightline', no_heuristic_straightline_stats, heuristic_probabilistic_straightline_stats, heuristic_deterministic_straightline_stats)

# if os.path.exists(f'{root_folder}/_results/{domain_name}_no_heuristic_deepreactive_probabilistic_statistics.pickle'):
#     no_heuristic_drp_stats = load_data(f'{root_folder}/_results/{domain_name}_no_heuristic_deepreactive_probabilistic_statistics.pickle')
#     heuristic_deterministic_drp_stats = load_data(f'{root_folder}/_results/{domain_name}_heuristic_deepreactive_deterministic_statistics.pickle')
#     heuristic_probabilistic_drp_stats = load_data(f'{root_folder}/_results/{domain_name}_heuristic_deepreactive_probabilistic_statistics.pickle')

#     plot_cost_curve_per_iteration(f'Best Costs per Iteration - DRP ({domain_name})', 'Best', 'best_returns', 'drp', no_heuristic_drp_stats, heuristic_probabilistic_drp_stats, lambda item : -item.best_return)
#     plot_cost_curve_per_iteration(f'Train Costs per Iteration - DRP ({domain_name})', 'Train', 'train_returns', 'drp', no_heuristic_drp_stats, heuristic_probabilistic_drp_stats, lambda item : -item.train_return)
#     plot_time_bars(f'Mean execution time - DRP ({domain_name})', 'drp', no_heuristic_drp_stats, heuristic_probabilistic_drp_stats, heuristic_deterministic_drp_stats)

print('done!')
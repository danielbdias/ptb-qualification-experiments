from _utils import load_data
from _domains import domains
from _graphs import plot_cost_curve_per_iteration

import numpy as np

import csv
import os

def write_statistic_header(file_path):
    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Domain', 'State fluent', 'Mean (Base)', 'Mean (Fluent)', 'Variance', 'Range'])

def write_statistic_line(domain, variable, base_statistics, fluent_statistics, file_path):
    best_returns_base_statistics = list(map(lambda item : item.statistics_history[-1].best_return, base_statistics))
    best_returns_fluent_statistics = list(map(lambda item : item.statistics_history[-1].best_return, fluent_statistics))

    best_returns_base_statistics_mean = np.mean(best_returns_base_statistics)
    best_returns_fluent_statistics_mean = np.mean(best_returns_fluent_statistics)
    best_returns_fluent_statistics_variance = np.var(best_returns_fluent_statistics)
    fluent_range = np.abs(best_returns_base_statistics_mean - best_returns_fluent_statistics_mean)

    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # writer.writerow(['Domain', 'State fluent', 'Mean (Base)', 'Mean (Fluent)', 'Variance', 'Range'])
        writer.writerow([domain, variable, best_returns_base_statistics_mean, best_returns_fluent_statistics_mean, best_returns_fluent_statistics_variance, fluent_range])

print('--------------------------------------------------------------------------------')
print('Generating graphs')
print('--------------------------------------------------------------------------------')
print()

root_folder = os.path.dirname(__file__)

statistic_file = f'{root_folder}/_plots/statistics_slp.csv'
write_statistic_header(statistic_file)

for domain in domains:
    domain_name = domain.name
    variables_removed = domain.variables_removed

    statistics = {
        'Regular': load_data(f'{root_folder}/_results/{domain_name}_regular_statistics.pickle')
    }

    for variable in variables_removed:
        statistics[f'Simplified without {variable}'] = load_data(f'{root_folder}/_results/{domain_name}_simplified_{variable}_statistics.pickle')

    graph_path = f'{root_folder}/_plots/{domain_name}_slp.pdf'
    plot_cost_curve_per_iteration(f'Best Costs per Iteration ({domain_name})', statistics, lambda item : -item.best_return, graph_path)

    base_statistics = statistics['Regular']

    for fluent in variables_removed:
        fluent_statistics = statistics[f'Simplified without {fluent}']
        write_statistic_line(domain.name, fluent, base_statistics, fluent_statistics, statistic_file)

print('done!')
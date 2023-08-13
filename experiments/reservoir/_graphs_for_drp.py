from _utils import load_data

import os
import numpy as np
import matplotlib.pyplot as plt

root_folder = os.path.dirname(__file__)
no_heuristic_stats_file = f'{root_folder}/probabilistic_no_heuristic_deepreactive_statistics.pickle'
heuristic_stats_file = f'{root_folder}/probabilistic_with_heuristic_deepreactive_statistics.pickle'

no_heuristic_stats = load_data(no_heuristic_stats_file)

no_heuristic_iterations = list(map(lambda item : item.iteration, no_heuristic_stats))
no_heuristic_best_returns = list(map(lambda item : float(item.best_return) * -1, no_heuristic_stats))

heuristic_stats = load_data(heuristic_stats_file)

heuristic_iterations = list(map(lambda item : item.iteration, heuristic_stats))
heuristic_best_returns = list(map(lambda item : float(item.best_return) * -1, heuristic_stats))

plt.subplots(1, figsize=(10,10))
plt.plot(no_heuristic_iterations, no_heuristic_best_returns, '--', label="No Heuristic")
plt.plot(heuristic_iterations, heuristic_best_returns, label="Heuristic")

# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

plt.title("Learning Curve - Straight line Plan")
plt.xlabel("Iterations"), plt.ylabel("Costs"), plt.legend(loc="best")
plt.tight_layout()

plt.savefig(f'{root_folder}/graph.png', format='png')
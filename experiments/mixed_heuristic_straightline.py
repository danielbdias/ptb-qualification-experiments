import jax.nn.initializers as initializers
import os
import sys
import numpy as np

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan

from _utils import run_experiment, save_data
from _common_params import get_planner_params, HEURISTIC_STRAIGHTLINE_ACTION, JAX_SEEDS

######################################################################################################################################################
# Script start
######################################################################################################################################################

root_folder = os.path.dirname(__file__)
domain_name = sys.argv[1]

mean_deterministic_domain_file=f'{root_folder}/{domain_name}/deterministic/domain.rddl'
mean_deterministic_instance_file=f'{root_folder}/{domain_name}/deterministic/instance0.rddl'

max_deterministic_domain_file=f'{root_folder}/{domain_name}/deterministic/domain_max.rddl'
max_deterministic_instance_file=f'{root_folder}/{domain_name}/deterministic/instance0.rddl'

probabilistic_domain_file=f'{root_folder}/{domain_name}/probabilistic/domain.rddl'
probabilistic_instance_file=f'{root_folder}/{domain_name}/probabilistic/instance0.rddl'

mean_deterministic_experiment_stats = []
max_deterministic_experiment_stats = []
probabilistic_experiment_stats = []

for jax_seed in JAX_SEEDS:
    
    
    # Step 1 - run planner with "deterministic" (mean) environment
    mean_deterministic_planner_parameters = get_planner_params(plan=JaxStraightLinePlan(), jax_seed=jax_seed, deterministic=True)
    mean_deterministic_environment = RDDLEnv.RDDLEnv(domain=mean_deterministic_domain_file, instance=mean_deterministic_instance_file)
    mean_deterministic_experiment_summary = run_experiment("Deterministic (Mean) - Straight line", environment=mean_deterministic_environment, planner_parameters=mean_deterministic_planner_parameters)
    mean_deterministic_experiment_stats.append(mean_deterministic_experiment_summary)

    # Step 2 - run planner with "deterministic" (max) environment
    max_deterministic_planner_parameters = get_planner_params(plan=JaxStraightLinePlan(), jax_seed=jax_seed, deterministic=True)
    max_deterministic_environment = RDDLEnv.RDDLEnv(domain=max_deterministic_domain_file, instance=max_deterministic_instance_file)
    max_deterministic_experiment_summary = run_experiment("Deterministic (Max) - Straight line", environment=max_deterministic_environment, planner_parameters=max_deterministic_planner_parameters)
    max_deterministic_experiment_stats.append(max_deterministic_experiment_summary)

    # Step 3 - combine heristics
    mean_policy_weights = mean_deterministic_experiment_summary.final_policy_weights[HEURISTIC_STRAIGHTLINE_ACTION]
    max_policy_weights = max_deterministic_experiment_summary.final_policy_weights[HEURISTIC_STRAIGHTLINE_ACTION]

    combined_policy_weights = []

    for i in range(len(mean_policy_weights)):
        action_values = []
        for j in range(len(mean_policy_weights[i])):
            action_values.append(np.mean([ mean_policy_weights[i][j], max_policy_weights[i][j] ]))
        
        combined_policy_weights.append(action_values)

    combined_policy_weights = np.array(combined_policy_weights)

    # Step 4 - run planner with probabilistic environment and weights from last step
    probabilistic_environment = RDDLEnv.RDDLEnv(domain=probabilistic_domain_file, instance=probabilistic_instance_file)
    probabilistic_planner_parameters = get_planner_params(plan=JaxStraightLinePlan(initializer=initializers.constant(combined_policy_weights)), jax_seed=jax_seed)
    probabilistic_experiment_summary = run_experiment("Probabilistic + Heuristic - Straight line", environment=probabilistic_environment, planner_parameters=probabilistic_planner_parameters)
    probabilistic_experiment_stats.append(probabilistic_experiment_summary)

# Save experiment statistics

save_data(mean_deterministic_experiment_stats, f'{root_folder}/_results/{domain_name}_mixed_heuristic_straightline_mean_deterministic_statistics.pickle')
save_data(max_deterministic_experiment_stats, f'{root_folder}/_results/{domain_name}_mixed_heuristic_straightline_max_deterministic_statistics.pickle')
save_data(probabilistic_experiment_stats, f'{root_folder}/_results/{domain_name}_mixed_heuristic_straightline_probabilistic_statistics.pickle')
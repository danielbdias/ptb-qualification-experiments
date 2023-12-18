import os
import jax
import optax

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan

from _utils import run_experiment, save_data, PlannerParameters

root_folder = os.path.dirname(__file__)

jax_seeds = [42, 967, 61, 647, 347, 139, 367, 13, 971, 31]

domain = 'HVAC'
instance = 'instance1'
action_bounds = {'fan-in': (0.05001, None), 'heat-input': (0.0, None)}

variables_removed = ['occupied', 'tempzone']

experiment_params = {
    'batch_size_train': 256,
    'optimizer': optax.rmsprop,
    'learning_rate': 0.1,
    'epochs': 1000,
    'action_bounds': action_bounds,
    'report_statistics_interval': 1,
    'epsilon_error': 0.001,
    'epsilon_iteration_stop': 10,
}

#########################################################################################################
# Runs with regular domain
#########################################################################################################

regular_environment = RDDLEnv.RDDLEnv(domain=f'{root_folder}/domains/{domain}/regular/domain.rddl', instance=f'{root_folder}/domains/{domain}/regular/{instance}.rddl')
regular_env_experiment_stats = []

for jax_seed in jax_seeds:
    experiment_params['plan'] = JaxStraightLinePlan()
    experiment_params['seed'] = jax.random.PRNGKey(jax_seed)

    regular_env_params = PlannerParameters(**experiment_params)

    regular_env_experiment_summary = run_experiment(f"{domain} Regular - Straight line", environment=regular_environment, planner_parameters=regular_env_params)
    regular_env_experiment_stats.append(regular_env_experiment_summary)

save_data(regular_env_experiment_stats, f'{root_folder}/_results/{domain}_regular_statistics.pickle')

#########################################################################################################
# Runs with simplified domains
#########################################################################################################

for variable in variables_removed:
    simplified_environment = RDDLEnv.RDDLEnv(domain=f'{root_folder}/domains/{domain}/simplified_{variable}/domain.rddl', instance=f'{root_folder}/domains/{domain}/simplified_{variable}/{instance}.rddl')
    simplified_env_experiment_stats = []
    heuristic_experiment_stats = []

    for jax_seed in jax_seeds:
        experiment_params['plan'] = JaxStraightLinePlan()
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)

        simplified_env_params = PlannerParameters(**experiment_params)

        simplified_env_experiment_summary = run_experiment(f"{domain} (without {variable}) - Straight line", environment=simplified_environment, planner_parameters=simplified_env_params)
        simplified_env_experiment_stats.append(simplified_env_experiment_summary)

    save_data(simplified_env_experiment_stats, f'{root_folder}/_results/{domain}_simplified_{variable}_statistics.pickle')
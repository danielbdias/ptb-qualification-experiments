import os
import jax
import optax
import time

from dataclasses import dataclass

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan

from _utils import run_experiment, save_data, PlannerParameters

@dataclass(frozen=True)
class DomainExperiment:
    name:              str
    instance:          str
    action_bounds:     dict
    variables_removed: list

root_folder = os.path.dirname(__file__)

jax_seeds = [42, 967, 61, 647, 347, 139, 367, 13, 971, 31]

silent = False

domains = [
    DomainExperiment(
        name='HVAC',
        instance='instance1',
        action_bounds={'fan-in': (0.05001, None), 'heat-input': (0.0, None)},
        variables_removed=['occupied', 'tempzone']
    ),
    DomainExperiment(
        name='UAV',
        instance='instance1',
        action_bounds={'set-acc': (-1, 1), 'set-phi': (-1, 1), 'set-theta': (-1, 1)},
        variables_removed=['pos-x', 'pos-y', 'pos-z']
    )
]

experiment_params = {
    'batch_size_train': 256,
    'optimizer': optax.rmsprop,
    'learning_rate': 0.1,
    'epochs': 1000,
    'report_statistics_interval': 1,
    'epsilon_error': 0.001,
    'epsilon_iteration_stop': 10,
}

start_time = time.time()

#########################################################################################################
# Runs with regular domain
#########################################################################################################

for domain in domains:
    regular_environment = RDDLEnv.RDDLEnv(domain=f'{root_folder}/domains/{domain.name}/regular/domain.rddl', instance=f'{root_folder}/domains/{domain.name}/regular/{domain.instance}.rddl')
    regular_env_experiment_stats = []

    for jax_seed in jax_seeds:
        experiment_params['plan'] = JaxStraightLinePlan()
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds

        regular_env_params = PlannerParameters(**experiment_params)

        regular_env_experiment_summary = run_experiment(f"{domain.name} Regular - Straight line", environment=regular_environment, planner_parameters=regular_env_params, silent=silent)
        regular_env_experiment_stats.append(regular_env_experiment_summary)

    save_data(regular_env_experiment_stats, f'{root_folder}/_results/{domain.name}_regular_statistics.pickle')

    #########################################################################################################
    # Runs with simplified domains
    #########################################################################################################

    for variable in domain.variables_removed:
        simplified_environment = RDDLEnv.RDDLEnv(domain=f'{root_folder}/domains/{domain.name}/simplified_{variable}/domain.rddl', instance=f'{root_folder}/domains/{domain.name}/simplified_{variable}/{domain.instance}.rddl')
        simplified_env_experiment_stats = []
        heuristic_experiment_stats = []

        for jax_seed in jax_seeds:
            experiment_params['plan'] = JaxStraightLinePlan()
            experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
            experiment_params['action_bounds'] = domain.action_bounds

            simplified_env_params = PlannerParameters(**experiment_params)

            simplified_env_experiment_summary = run_experiment(f"{domain.name} (without {variable}) - Straight line", environment=simplified_environment, planner_parameters=simplified_env_params, silent=silent)
            simplified_env_experiment_stats.append(simplified_env_experiment_summary)

        save_data(simplified_env_experiment_stats, f'{root_folder}/_results/{domain.name}_simplified_{variable}_statistics.pickle')

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()
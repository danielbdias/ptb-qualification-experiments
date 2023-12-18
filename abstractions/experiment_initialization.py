import os
import sys
import jax
import optax
import jax.nn.initializers as initializers

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan

from _utils import run_experiment, save_data, PlannerParameters

root_folder = os.path.dirname(__file__)

jax_seeds = [42, 967, 61, 647, 347, 139, 367, 13, 971, 31]

domain = 'HVAC'
instance = 'instance1'
action_bounds = {'fan-in': (0.05001, None), 'heat-input': (0.0, None)}

# domain = 'RaceCar'
# instance = 'instance0'
# action_bounds={'fx': (-1, 1), 'fy': (-1, 1)}

#########################################################################################################
# Runs with regular domain
#########################################################################################################

regular_environment = RDDLEnv.RDDLEnv(domain=f'{root_folder}/domains/{domain}/regular/domain.rddl', instance=f'{root_folder}/domains/{domain}/regular/{instance}.rddl')
regular_env_experiment_stats = []

for jax_seed in jax_seeds:
    regular_env_params = PlannerParameters(
        batch_size_train=256,
        plan=JaxStraightLinePlan(),
        optimizer=optax.rmsprop,
        learning_rate=0.1,
        epochs=1000,
        seed=jax.random.PRNGKey(jax_seed),
        action_bounds=action_bounds,
        report_statistics_interval=1,
        epsilon_error=0.001,
        epsilon_iteration_stop=10,
    )

    regular_env_experiment_summary = run_experiment(f"{domain} Regular - Straight line", environment=regular_environment, planner_parameters=regular_env_params)
    regular_env_experiment_stats.append(regular_env_experiment_summary)

save_data(regular_env_experiment_stats, f'{root_folder}/_results/{domain}_regular_statistics.pickle')

#########################################################################################################
# Runs with simplified domain
#########################################################################################################

simplified_environment = RDDLEnv.RDDLEnv(domain=f'{root_folder}/domains/{domain}/simplified/domain.rddl', instance=f'{root_folder}/domains/{domain}/simplified/{instance}.rddl')
simplified_env_experiment_stats = []
heuristic_experiment_stats = []

for jax_seed in jax_seeds:
    simplified_env_params = PlannerParameters(
        batch_size_train=32,
        plan=JaxStraightLinePlan(),
        optimizer=optax.rmsprop,
        learning_rate=0.1,
        epochs=1000,
        seed=jax.random.PRNGKey(jax_seed),
        action_bounds=action_bounds,
        report_statistics_interval=1,
        epsilon_error=0.001,
        epsilon_iteration_stop=10,
    )

    simplified_env_experiment_summary = run_experiment(f"{domain} Simplified - Straight line", environment=simplified_environment, planner_parameters=simplified_env_params)
    simplified_env_experiment_stats.append(simplified_env_experiment_summary)

    initializers_per_action = {}
    for key in simplified_env_experiment_summary.final_policy_weights.keys():
        initializers_per_action[key] = initializers.constant(simplified_env_experiment_summary.final_policy_weights[key])

    heuristic_env_params = PlannerParameters(
        batch_size_train=256,
        plan=JaxStraightLinePlan(initializer_per_action=initializers_per_action),
        optimizer=optax.rmsprop,
        learning_rate=0.1,
        epochs=1000,
        seed=jax.random.PRNGKey(jax_seed),
        action_bounds=action_bounds,
        report_statistics_interval=1,
        epsilon_error=0.001,
        epsilon_iteration_stop=10,
    )

    heuristic_experiment_summary = run_experiment(f"{domain} Heuristic - Straight line", environment=regular_environment, planner_parameters=heuristic_env_params)
    heuristic_experiment_stats.append(heuristic_experiment_summary)

save_data(simplified_env_experiment_stats, f'{root_folder}/_results/{domain}_simplified_statistics.pickle')
save_data(heuristic_experiment_stats, f'{root_folder}/_results/{domain}_heuristic_statistics.pickle')
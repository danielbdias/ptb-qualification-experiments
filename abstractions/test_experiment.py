import os
import sys
import jax
import optax

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan

from _utils import run_experiment, save_data, PlannerParameters

root_folder = os.path.dirname(__file__)

# jax_seeds = [42, 967, 61, 647, 347, 139, 367, 13, 971, 31]
jax_seeds = [42, 967, 61, 647]

#########################################################################################################
# Runs with regular domain
#########################################################################################################

regular_environment = RDDLEnv.RDDLEnv(domain=f'{root_folder}/domains/UAV/regular/domain.rddl', instance=f'{root_folder}/domains/UAV/regular/instance1.rddl')
regular_env_experiment_stats = []

for jax_seed in jax_seeds:
    regular_env_params = PlannerParameters(
        batch_size_train=256,
        plan=JaxStraightLinePlan(),
        optimizer=optax.rmsprop,
        learning_rate=0.1,
        epochs=1000,
        seed=jax.random.PRNGKey(jax_seed),
        action_bounds={'set-acc': (-1, 1), 'set-phi': (-1, 1), 'set-theta': (-1, 1)},
        report_statistics_interval=1,
        epsilon_error=0.001,
        epsilon_iteration_stop=10,
    )

    regular_env_experiment_summary = run_experiment("UAV Regular - Straight line", environment=regular_environment, planner_parameters=regular_env_params)
    regular_env_experiment_stats.append(regular_env_experiment_summary)

save_data(regular_env_experiment_stats, f'{root_folder}/_results/UAV_regular_statistics.pickle')

#########################################################################################################
# Runs with simplified domain
#########################################################################################################

simplified_environment = RDDLEnv.RDDLEnv(domain=f'{root_folder}/domains/UAV/simplified/domain.rddl', instance=f'{root_folder}/domains/UAV/simplified/instance1.rddl')
simplified_env_experiment_stats = []

for jax_seed in jax_seeds:
    simplified_env_params = PlannerParameters(
        batch_size_train=256,
        plan=JaxStraightLinePlan(),
        optimizer=optax.rmsprop,
        learning_rate=0.1,
        epochs=1000,
        seed=jax.random.PRNGKey(jax_seed),
        action_bounds={'set-acc': (-1, 1), 'set-phi': (-1, 1), 'set-theta': (-1, 1)},
        report_statistics_interval=1,
        epsilon_error=0.001,
        epsilon_iteration_stop=10,
    )

    simplified_env_experiment_summary = run_experiment("UAV Simplified - Straight line", environment=simplified_environment, planner_parameters=simplified_env_params)
    simplified_env_experiment_stats.append(simplified_env_experiment_summary)

save_data(simplified_env_experiment_stats, f'{root_folder}/_results/UAV_simplified_statistics.pickle')
import os
import jax

import time

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan

from _domains import domains, jax_seeds, silent, experiment_params
from _utils import run_experiment, save_data, PlannerParameters

root_folder = os.path.dirname(__file__)

start_time = time.time()

#########################################################################################################
# Runs with regular domain
#########################################################################################################

for domain in domains:
    if not silent:
        print('--------------------------------------------------------------------------------')
        print('Domain: ', domain)
        print('--------------------------------------------------------------------------------')
        print()

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
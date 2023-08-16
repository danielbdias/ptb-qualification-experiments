import os

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxDeepReactivePolicy

from _utils import run_experiment, run_planner, save_data, save_time
from _common_params import get_planner_params, NETWORK_TOPOLOGY, JAX_SEEDS

######################################################################################################################################################
# Script start
######################################################################################################################################################

root_folder = os.path.dirname(__file__)

deterministic_domain_file=f'{root_folder}/deterministic/domain.rddl'
deterministic_instance_file=f'{root_folder}/deterministic/instance0.rddl'

probabilistic_domain_file=f'{root_folder}/probabilistic/domain.rddl'
probabilistic_instance_file=f'{root_folder}/probabilistic/instance0.rddl'

deterministic_experiment_stats = []
probabilistic_experiment_stats = []

for jax_seed in JAX_SEEDS:
    # Step 1 - run planner with "deterministic" environment
    deterministic_environment = RDDLEnv.RDDLEnv(domain=deterministic_domain_file, instance=deterministic_instance_file)
    deterministic_planner_parameters = get_planner_params(plan=JaxDeepReactivePolicy(topology=NETWORK_TOPOLOGY, jax_seed=jax_seed), drp=True)
    deterministic_experiment_summary = run_experiment("Deterministic - DRP", run_planner, environment=deterministic_environment, planner_parameters=deterministic_planner_parameters)
    deterministic_experiment_stats.append(deterministic_experiment_summary)

    # Step 2 - run planner with probabilistic environment and weights from last step
    probabilistic_environment = RDDLEnv.RDDLEnv(domain=probabilistic_domain_file, instance=probabilistic_instance_file)
    probabilistic_planner_parameters = get_planner_params(plan=JaxDeepReactivePolicy(topology=NETWORK_TOPOLOGY, weights_per_layer=deterministic_experiment_summary.final_policy_weights), drp=True, jax_seed=jax_seed)
    probabilistic_experiment_summary = run_experiment("Probabilistic + Heuristic - DRP", run_planner, environment=probabilistic_environment, planner_parameters=probabilistic_planner_parameters)
    probabilistic_experiment_stats.append(probabilistic_experiment_summary)

# Save experiment statistics

save_data(deterministic_experiment_stats, f'{root_folder}/zzz_heuristic_deepreactive_deterministic_statistics.pickle')
save_data(probabilistic_experiment_stats, f'{root_folder}/zzz_heuristic_deepreactive_probabilistic_statistics.pickle')
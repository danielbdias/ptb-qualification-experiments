import os

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxDeepReactivePolicy

from _utils import run_experiment, run_planner, save_data, save_time
from _common_params import get_planner_params, NETWORK_TOPOLOGY, JAX_SEEDS

root_folder = os.path.dirname(__file__)
probabilistic_domain_file=f'{root_folder}/probabilistic/domain.rddl'
probabilistic_instance_file=f'{root_folder}/probabilistic/instance0.rddl'

probabilistic_experiment_stats = []

for jax_seed in JAX_SEEDS:
    probabilistic_environment = RDDLEnv.RDDLEnv(domain=probabilistic_domain_file, instance=probabilistic_instance_file)
    probabilistic_planner_parameters = get_planner_params(plan=JaxDeepReactivePolicy(topology=NETWORK_TOPOLOGY), drp=True, jax_seed=jax_seed)
    probabilistic_experiment_summary = run_experiment("Probabilistic (no heuristic) - DRP", run_planner, environment=probabilistic_environment, planner_parameters=probabilistic_planner_parameters)
    probabilistic_experiment_stats.append(probabilistic_experiment_summary)

save_data(probabilistic_experiment_stats, f'{root_folder}/zzz_no_heuristic_deepreactive_probabilistic_statistics.pickle')
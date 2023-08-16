import jax.nn.initializers as initializers
import os

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan

from _utils import run_experiment, run_planner, save_data, save_time
from _common_params import get_planner_params, HEURISTIC_STRAIGHTLINE_ACTION, JAX_SEEDS

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
    deterministic_planner_parameters = get_planner_params(plan=JaxStraightLinePlan(), jax_seed=jax_seed)
    deterministic_experiment_summary = run_experiment("Deterministic - Straight line", run_planner, environment=deterministic_environment, planner_parameters=deterministic_planner_parameters)
    deterministic_experiment_stats.append(deterministic_experiment_summary)

    # Step 2 - run planner with probabilistic environment and weights from last step
    probabilistic_environment = RDDLEnv.RDDLEnv(domain=probabilistic_domain_file, instance=probabilistic_instance_file)
    probabilistic_planner_parameters = get_planner_params(plan=JaxStraightLinePlan(initializer=initializers.constant(deterministic_experiment_summary.final_policy_weights[HEURISTIC_STRAIGHTLINE_ACTION])), jax_seed=jax_seed)
    probabilistic_experiment_summary = run_experiment("Probabilistic + Heuristic - Straight line", run_planner, environment=probabilistic_environment, planner_parameters=probabilistic_planner_parameters)
    probabilistic_experiment_stats.append(probabilistic_experiment_summary)

# Save experiment statistics

save_data(deterministic_experiment_stats, f'{root_folder}/zzz_heuristic_straightline_deterministic_statistics.pickle')
save_data(probabilistic_experiment_stats, f'{root_folder}/zzz_heuristic_straightline_probabilistic_statistics.pickle')
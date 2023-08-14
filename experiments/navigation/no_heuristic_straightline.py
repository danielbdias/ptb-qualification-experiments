import os

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan

from _utils import run_experiment, run_planner, save_data, save_time
from _common_params import get_planner_params

root_folder = os.path.dirname(__file__)

# specify the model
probabilistic_domain_file=f'{root_folder}/probabilistic/domain.rddl'
probabilistic_instance_file=f'{root_folder}/probabilistic/instance0.rddl'
probabilistic_environment = RDDLEnv.RDDLEnv(domain=probabilistic_domain_file, instance=probabilistic_instance_file)

probabilistic_planner_parameters = get_planner_params(plan=JaxStraightLinePlan())

probabilistic_final_policy_weights, probabilistic_statistics_history, experiment_time = run_experiment("Probabilistic (no heuristic) - Straight line", run_planner, environment=probabilistic_environment, planner_parameters=probabilistic_planner_parameters)
save_data(probabilistic_final_policy_weights, f'{root_folder}/zzz_probabilistic_no_heuristic_straightline_policy.pickle')
save_data(probabilistic_statistics_history, f'{root_folder}/zzz_probabilistic_no_heuristic_straightline_statistics.pickle')
save_time("Probabilistic (no heuristic) - Straight line", experiment_time, f'{root_folder}/zzz_straightline_time.csv')
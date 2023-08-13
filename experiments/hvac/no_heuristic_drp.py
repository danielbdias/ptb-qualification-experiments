
import jax
import optax
import os

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxDeepReactivePolicy

from _utils import PlannerParameters, run_experiment, run_planner, save_data

root_folder = os.path.dirname(__file__)

# specify the model
probabilistic_domain_file=f'{root_folder}/probabilistic/domain.rddl'
probabilistic_instance_file=f'{root_folder}/probabilistic/instance0.rddl'
probabilistic_environment = RDDLEnv.RDDLEnv(domain=probabilistic_domain_file, instance=probabilistic_instance_file)

probabilistic_planner_parameters = PlannerParameters(
    batch_size_train=32,
    plan=JaxDeepReactivePolicy(topology=[256, 128, 64, 32]),
    optimizer=optax.rmsprop,
    learning_rate=0.1,
    epochs=1000,
    seed=jax.random.PRNGKey(42),
    action_bounds={'air':(0.0, 10.0)},
    report_statistics_interval=10
)

probabilistic_final_policy_weights, probabilistic_statistics_history = run_experiment("Probabilistic (no heuristic) - DRP", run_planner, environment=probabilistic_environment, planner_parameters=probabilistic_planner_parameters)
save_data(probabilistic_final_policy_weights, f'{root_folder}/zzz_probabilistic_no_heuristic_deepreactive_policy.pickle')
save_data(probabilistic_statistics_history, f'{root_folder}/zzz_probabilistic_no_heuristic_deepreactive_statistics.pickle')

import jax
import optax

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan

from _utils import PlannerParameters, run_experiment, run_planner, save_data

# specify the model
domain_file='/Users/danielbdias/Development/Repositories/studies/doctorate/experiments/planning-through-backpropagation/domains/probabilistic_navigation/domain.rddl'
probabilistic_instance_file='/Users/danielbdias/Development/Repositories/studies/doctorate/experiments/planning-through-backpropagation/domains/probabilistic_navigation/instance0.rddl'
probabilistic_environment = RDDLEnv.RDDLEnv(domain=domain_file, instance=probabilistic_instance_file)

probabilistic_planner_parameters = PlannerParameters(
    batch_size_train=128,
    plan=JaxStraightLinePlan(),
    optimizer=optax.rmsprop,
    learning_rate=0.1,
    epochs=1000,
    seed=jax.random.PRNGKey(42),
    report_statistics_interval=10
)

probabilistic_final_policy_weights, probabilistic_statistics_history = run_experiment("Probabilistic (no heuristic) - Straight line", run_planner, environment=probabilistic_environment, planner_parameters=probabilistic_planner_parameters)
save_data(probabilistic_final_policy_weights, '/Users/danielbdias/Development/Repositories/studies/doctorate/experiments/planning-through-backpropagation/experiment_data/probabilistic_no_heuristic_straightline_policy.pickle')
save_data(probabilistic_statistics_history, '/Users/danielbdias/Development/Repositories/studies/doctorate/experiments/planning-through-backpropagation/experiment_data/probabilistic_no_heuristic_straightline_statistics.pickle')
print(probabilistic_final_policy_weights)
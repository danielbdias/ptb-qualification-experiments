import haiku as hk
import jax
import optax

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxDeepReactivePolicy

from _utils import PlannerParameters, run_experiment, run_planner, save_data

######################################################################################################################################################
# Script start
######################################################################################################################################################

# specify the model

# Step 1 - run planner with "deterministic" environment
deterministic_domain_file='/Users/danielbdias/Development/Repositories/studies/doctorate/experiments/planning-through-backpropagation/domains/deterministic_navigation/domain.rddl'
deterministic_instance_file='/Users/danielbdias/Development/Repositories/studies/doctorate/experiments/planning-through-backpropagation/domains/deterministic_navigation/instance0.rddl'
deterministic_environment = RDDLEnv.RDDLEnv(domain=deterministic_domain_file, instance=deterministic_instance_file)

deterministic_planner_parameters = PlannerParameters(
    batch_size_train=32,
    plan=JaxDeepReactivePolicy(topology=[256, 128, 64, 32]),
    optimizer=optax.rmsprop,
    learning_rate=0.1,
    epochs=1000,
    seed=jax.random.PRNGKey(42),
    report_statistics_interval=10
)

deterministic_final_policy_weights, deterministic_statistics_history = run_experiment("Deterministic - DRP", run_planner, environment=deterministic_environment, planner_parameters=deterministic_planner_parameters)
save_data(deterministic_final_policy_weights, '/Users/danielbdias/Development/Repositories/studies/doctorate/experiments/planning-through-backpropagation/experiment_data/deterministic_deepreactive_policy.pickle')
save_data(deterministic_statistics_history, '/Users/danielbdias/Development/Repositories/studies/doctorate/experiments/planning-through-backpropagation/experiment_data/deterministic_deepreactive_statistics.pickle')

# Step 2 - run planner with probabilistic environment and weights from last step
deterministic_domain_file='/Users/danielbdias/Development/Repositories/studies/doctorate/experiments/planning-through-backpropagation/domains/probabilistic_navigation/domain.rddl'
probabilistic_instance_file='/Users/danielbdias/Development/Repositories/studies/doctorate/experiments/planning-through-backpropagation/domains/probabilistic_navigation/instance0.rddl'
probabilistic_environment = RDDLEnv.RDDLEnv(domain=domain_file, instance=probabilistic_instance_file)

probabilistic_planner_parameters = PlannerParameters(
    batch_size_train=32,
    plan=JaxDeepReactivePolicy(topology=[256, 128, 64, 32], weights_per_layer=deterministic_final_policy_weights),
    optimizer=optax.rmsprop,
    learning_rate=0.1,
    epochs=1000,
    seed=jax.random.PRNGKey(42),
    report_statistics_interval=10
)

probabilistic_final_policy_weights, probabilistic_statistics_history = run_experiment("Probabilistic + Heuristic - DRP", run_planner, environment=probabilistic_environment, planner_parameters=probabilistic_planner_parameters)
save_data(probabilistic_final_policy_weights, '/Users/danielbdias/Development/Repositories/studies/doctorate/experiments/planning-through-backpropagation/experiment_data/probabilistic_with_heuristic_deepreactive_policy.pickle')
save_data(probabilistic_statistics_history, '/Users/danielbdias/Development/Repositories/studies/doctorate/experiments/planning-through-backpropagation/experiment_data/probabilistic_with_heuristic_deepreactive_statistics.pickle')
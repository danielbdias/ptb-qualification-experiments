import jax
import optax

from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner, JaxDeepReactivePolicy

from dataclasses import dataclass
import time
import pickle

@dataclass(frozen=True)
class PlannerParameters:
    batch_size_train:           int
    plan:                       JaxDeepReactivePolicy
    optimizer:                  optax.GradientTransformation
    learning_rate:              float
    epochs:                     int
    seed:                       jax.random.KeyArray
    report_statistics_interval: int
    action_bounds:              dict

@dataclass(frozen=True)
class ExperimentStatistics:
    iteration:     int
    train_return:  float
    test_return:   float
    best_return:   float
    best_params:   dict
    last_callback: bool

    @staticmethod
    def from_callback(planner_callback, total_epochs):
        iteration = planner_callback['iteration']

        last_callback = (iteration >= total_epochs)

        # possible keys: 'iteration', 'train_return', 'test_return', 'best_return', 'params', 'best_params', 'last_iteration_improved', 'grad', 'updates', 'action', 'error', 'invariant', 'precondition', 'pvar', 'reward', 'termination'
        best_params = dict()
        if last_callback:
            best_params = planner_callback['best_params']

        return ExperimentStatistics(
            iteration=iteration,
            train_return=planner_callback['train_return'],
            test_return=planner_callback['test_return'],
            best_return=planner_callback['best_return'],
            best_params=best_params,
            last_callback=last_callback,
        )
    
    def __str__(self) -> str:
        return 'step={} train_return={:.6f} test_return={:.6f} best_return={:.6f}'.format(
          str(self.iteration).rjust(4), self.train_return, self.test_return, self.best_return)
    
def run_experiment(name, experiment_action, **kwargs):
    print('--------------------------------------------------------------------------------')
    print('Experiment: ', name)
    print('--------------------------------------------------------------------------------')
    print()
    
    # Start timer
    start_time = time.time()

    # Code to be timed
    results = experiment_action(**kwargs)

    # End timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print('Elapsed time: {} seconds'.format(elapsed_time))
    print()

    return results

def run_planner(environment, planner_parameters):
    # initialize the planner
    planner = JaxRDDLBackpropPlanner(
        environment.model,
        batch_size_train=planner_parameters.batch_size_train,
        plan=planner_parameters.plan,
        optimizer=planner_parameters.optimizer,
        optimizer_kwargs={'learning_rate': planner_parameters.learning_rate},
        action_bounds=planner_parameters.action_bounds)

    # run the planner as an optimization process
    planner_callbacks = planner.optimize(
        planner_parameters.seed, 
        epochs=planner_parameters.epochs, 
        step=planner_parameters.report_statistics_interval
    )

    final_policy_weights = None
    statistics_history = []

    for callback in planner_callbacks:
        statistics = ExperimentStatistics.from_callback(callback, total_epochs=(planner_parameters.epochs-1))
        statistics_history.append(statistics)

        print(statistics)

        if statistics.last_callback:
            final_policy_weights = statistics.best_params

    return final_policy_weights, statistics_history

def save_data(data, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)
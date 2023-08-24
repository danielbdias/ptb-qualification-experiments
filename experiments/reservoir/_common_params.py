import jax
import optax

from _utils import PlannerParameters

def get_planner_params(plan, jax_seed, drp=False):
    learning_rate = LEARNING_RATE_STRAIGHTLINE
    if drp:
        learning_rate = LEARNING_RATE_DRP

    return PlannerParameters(
        batch_size_train=256,
        plan=plan,
        optimizer=optax.rmsprop,
        learning_rate=learning_rate,
        epochs=500,
        seed=jax.random.PRNGKey(jax_seed),
        action_bounds={'release':(0.0, 100.0)},
        report_statistics_interval=1,
        epsilon_error=0.001,
        epsilon_iteration_stop=10,
    )

LEARNING_RATE_STRAIGHTLINE=0.1
LEARNING_RATE_DRP=0.001
NETWORK_TOPOLOGY = [256, 128, 64, 32]
JAX_SEEDS = [42, 967, 61, 647, 347, 139, 367, 13, 971, 31]
HEURISTIC_STRAIGHTLINE_ACTION = 'outflow'
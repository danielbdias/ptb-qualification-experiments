import jax
import optax

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan

# specify the model
domain_file='/Users/danielbdias/Development/Repositories/studies/doctorate/experiments/planning-through-backpropagation/domains/probabilistic_navigation/domain.rddl'
probabilistic_instance_file='/Users/danielbdias/Development/Repositories/studies/doctorate/experiments/planning-through-backpropagation/domains/probabilistic_navigation/instance0.rddl'
myEnv = RDDLEnv.RDDLEnv(domain=domain_file, instance=probabilistic_instance_file)
model = myEnv.model

# initialize the planner
planner = JaxRDDLBackpropPlanner(
    model,
    batch_size_train=32,
    plan=JaxStraightLinePlan(),
    optimizer=optax.rmsprop,
    optimizer_kwargs={'learning_rate': 0.1})

for callback in planner.optimize(
    jax.random.PRNGKey(42), epochs=1000, step=10):
    print('step={} train_return={:.6f} test_return={:.6f} best_return={:.6f}'.format(
          str(callback['iteration']).rjust(4),
          callback['train_return'],
          callback['test_return'],
          callback['best_return']))
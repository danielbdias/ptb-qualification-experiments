import jax
import optax

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan

# specify the model
EnvInfo = ExampleManager.GetEnvInfo('Wildfire')
myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(0))
model = myEnv.model

# initialize the planner
planner = JaxRDDLBackpropPlanner(
    model,
    batch_size_train=32,
    plan=JaxStraightLinePlan(),
    optimizer=optax.rmsprop,
    optimizer_kwargs={'learning_rate': 0.1})

# train for 1000 epochs using gradient ascent - print progress every 50
# note that boolean actions are wrapped with sigmoid by default, so the
# policy_hyperparams dictionary must be filled with weights for them
policy_weights = {'cut-out': 10.0, 'put-out': 10.0}
for callback in planner.optimize(
    jax.random.PRNGKey(42), epochs=1000, step=10, policy_hyperparams=policy_weights):
    print('step={} train_return={:.6f} test_return={:.6f} best_return={:.6f}'.format(
          str(callback['iteration']).rjust(4),
          callback['train_return'],
          callback['test_return'],
          callback['best_return']))
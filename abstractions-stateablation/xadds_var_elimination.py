import os

from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Parser.parser import RDDLParser

# Read the domain and instance files
domain, instance = 'Reservoir_continuous', '0'
env_info = ExampleManager.GetEnvInfo(domain)
domain = env_info.get_domain()
instance = env_info.get_instance(instance)

# Read and parse domain and instance
reader = RDDLReader(domain, instance)
domain = reader.rddltxt
parser = RDDLParser(None, False)
parser.build()

# Parse RDDL file
rddl_ast = parser.parse(domain)

# Ground domain
grounder = RDDLGrounder(rddl_ast)
model = grounder.Ground()

model_xadd = RDDLModelWXADD(model)
model_xadd.compile()

print('Reward XADD variables:')
print(model_xadd._context.collect_vars(model_xadd.reward))
print()

print('Reward XADD:')
model_xadd.print(model_xadd.reward)

variable = 'overflow___t1'
print(f'Test for {variable}')

# dec, _ = model_xadd._context.get_dec_expr_index(variable, create=False)
# print(dec)

# print('Reduced XADD:')
# result = model_xadd._context.min_or_max_var(model_xadd.reward, 'overflow___t1', is_min=True)
# model_xadd.print(result)

# model_xadd.print(model_xadd.reward)
# model_xadd._context.save_graph(model_xadd.reward, file_name='reward')

# eliminate vars from XADD


## TODO
# plot reward function in 3d
# think about variable elimination
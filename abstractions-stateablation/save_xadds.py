import os

from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Parser.parser import RDDLParser

# Read the domain and instance files
domain, instance = 'HVAC', '0'
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

print('Variables:', list(model_xadd.cpfs.keys()))
print()

model_xadd.print(model_xadd.reward)
model_xadd._context.save_graph(model_xadd.reward, file_name='reward')

# for cpf_key in model_xadd.cpfs.keys():
#     cpf_structure = model_xadd.cpfs[cpf_key]
#     # model_xadd.print(cpf_structure)
#     model_xadd._context.save_graph(cpf_structure, file_name=cpf_key)
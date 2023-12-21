from pyRDDLGym.Visualizer.visualize_dbn import RDDL2Graph

# instances = [0, 1]

# for instance in instances:
#     r2g_UAV = RDDL2Graph(
#         domain='UAV_continuous',
#         instance=instance,
#         directed=True,
#         strict_grouping=True,
#     )

#     r2g_UAV.save_dbn(file_name='UAV')
#     # r2g.save_dbn(file_name='Wildfire', fluent='burning', gfluent='x1_y1')

# r2g_Reservoir = RDDL2Graph(
#     domain='Reservoir_continuous',
#     instance=0,
#     directed=True,
#     strict_grouping=True,
# )

# r2g_Reservoir.save_dbn(file_name='Reservoir')

# r2g_MarsRover = RDDL2Graph(
#     domain='MarsRover',
#     instance=0,
#     directed=True,
#     strict_grouping=True,
# )

# r2g_MarsRover.save_dbn(file_name='MarsRover')

r2g_RaceCar = RDDL2Graph(
    domain='RaceCar',
    instance=0,
    directed=True,
    strict_grouping=True,
)

r2g_RaceCar.save_dbn(file_name='RaceCar')
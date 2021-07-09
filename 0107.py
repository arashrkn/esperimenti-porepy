''' Prova Darcy con RT0 '''

import numpy as np
import porepy as pp

g = pp.CartGrid([20, 20], [1,1])
g.compute_geometry()

b_faces = g.tags['domain_boundary_faces'].nonzero()[0]
tipi_bc = np.full(b_faces.size, 'dir'); tipi_bc[0:20] = 'neu'
bc = pp.BoundaryCondition(g, b_faces, tipi_bc)
bc_val = np.zeros(g.num_faces)

parameters = {"bc": bc, "bc_values": bc_val}

data = pp.initialize_default_data(g, {}, "flow", parameters)

flow_discretization = pp.Tpfa("flow")
flow_discretization.discretize(g, data)
print('TPFA ok')

flow_discretization = pp.MVEM("flow")
flow_discretization.discretize(g, data)
print('MVEM ok')

flow_discretization = pp.RT0("flow")
flow_discretization.discretize(g, data)
print('RT0 ok')

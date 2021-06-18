# Darcy con framework AD

import numpy as np
import porepy as pp
import scipy.sparse.linalg as sps

gb = pp.meshing.cart_grid([], [10, 10], physdims=[1,1])

g,d = [(g,d) for g,d in gb][0]

d[pp.PRIMARY_VARIABLES] = {'pressure': {'cells': 1}}

permeabilita = pp.SecondOrderTensor(np.ones(g.num_cells))

facce_bordo = g.tags['domain_boundary_faces'].nonzero()[0]
valori_bc = np.zeros(g.num_faces)
valori_bc[facce_bordo[g.face_centers[0, facce_bordo] == 0]] = 1

tipi_bc = np.array(['neu']*facce_bordo.size)
tipi_bc[g.face_centers[0, facce_bordo] == 0] = 'dir'
tipi_bc[g.face_centers[0, facce_bordo] == 1] = 'dir'
bc = pp.BoundaryCondition(g, facce_bordo, tipi_bc)

parameters = { "second_order_tensor": permeabilita, "bc": bc, "bc_values": valori_bc } 
pp.initialize_data(g, d, 'flow', parameters)
pp.set_state(d)

div = pp.ad.Divergence([g])
bound_ad = pp.ad.BoundaryCondition('flow', grids=[g])

dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)

p = equation_manager.merge_variables([(g, 'pressure')])
d[pp.STATE]['pressure'] =  np.zeros(g.cell_centers.shape[1])

mpfa = pp.ad.MpfaAd('flow', [g])

# Nota: Gli step che devo prendere per vedere dei numeri veri e propri.
# interior_flux = mpfa.flux * p
# interior_flux_exp = pp.ad.Expression(interior_flux, dof_manager)
# interior_flux_exp.discretize(gb)
# interior_flux_num = interior_flux_exp.to_ad(gb=gb)

# Se gli operatori sono gia' stati discretizzati invece
# full_flux = interior_flux + mpfa.bound_flux * bound_ad
# full_flux_num = pp.ad.Expression(full_flux, dof_manager).to_ad(gb)

conservation = div * (mpfa.flux * p + mpfa.bound_flux * bound_ad)
conservation_exp = pp.ad.Expression(conservation, dof_manager, name='conservation')
conservation_exp.discretize(gb)
c = conservation_exp.to_ad(gb)

equation_manager.equations += [conservation_exp]
A, b = equation_manager.assemble_matrix_rhs()
solution = sps.spsolve(A, b)
dof_manager.distribute_variable(solution, additive=True)

pp.plot_grid(gb, 'pressure')

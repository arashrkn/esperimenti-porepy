'''
Colonna 1d.
Evolvo concentrazione di un soluto (trasporto e reazione) e di un precipitato (sola reazione).
Sia trasporto che reazione impliciti.
'''

import numpy as np
import porepy as pp
from porepy.numerics.ad.operators import Scalar, Array
import scipy.sparse.linalg as sps

gb = pp.meshing.cart_grid([], [30, 1], physdims=[10, 1])
exporter = pp.Exporter(gb, file_name='soluzione', folder_name='out/1906_2/')
g, d = [(g, d) for g, d in gb][0]


''' DATI, CONDIZIONI AL BORDO, CONDIZIONI INIZIALI '''
d[pp.PRIMARY_VARIABLES] = {'soluto': {'cells': 1}, 'precipitato': {'cells': 1}}
d[pp.STATE] = {}

TAU_R = 10
B = 0.8
DT = 0.5
T = 75

''' soluto '''
facce_bordo = g.tags["domain_boundary_faces"].nonzero()[0]

valori_bc = np.zeros(g.num_faces)
# valori_bc[facce_bordo[g.face_centers[0, facce_bordo] == 0]] = 1

tipi_bc = np.full(facce_bordo.size, 'dir')
tipi_bc[g.face_centers[1, facce_bordo] == 0] = 'neu'
tipi_bc[g.face_centers[1, facce_bordo] == 1] = 'neu'
bc = pp.BoundaryCondition(g, facce_bordo, tipi_bc)
# Nota: per controllare le bc: np.vstack((facce_bordo,tipi_bc)).transpose()

porosita = np.ones(g.num_cells)

darcy = np.zeros(g.num_faces)
darcy[g.face_normals[0, :] == 1] = 0.2

parametri_trasporto = {"bc": bc, "bc_values": valori_bc, "mass_weight": porosita, "darcy_flux": darcy}
pp.initialize_default_data(g, d, 'transport', parametri_trasporto)

d[pp.STATE]['soluto'] = np.zeros(g.num_cells)
d[pp.STATE]['soluto_0'] = np.zeros(g.num_cells)
d[pp.STATE]['soluto_0'][:] = 0

''' precipitato '''
d[pp.STATE]['precipitato'] = np.zeros(g.num_cells)
d[pp.STATE]['precipitato_0'] = np.zeros(g.num_cells)
d[pp.STATE]['precipitato_0'][g.cell_centers[0, :] <= 5] = 1
# d[pp.STATE]['precipitato_0'][:] = 1


''' EQUAZIONI '''
dof = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof)
soluto = equation_manager.merge_variables([(g, 'soluto')])
precipitato = equation_manager.merge_variables([(g, 'precipitato')])
soluto_0 = Array(d[pp.STATE]['soluto_0'])
precipitato_0 = Array(d[pp.STATE]['precipitato_0'])

div = pp.ad.Divergence([g])
bound_ad = pp.ad.BoundaryCondition('transport', grids=[g])
upwind = pp.ad.UpwindAd('transport', [g])
massa = pp.ad.MassMatrixAd('transport', [g])

r = 0.5*precipitato * (Scalar(1) - (1/B**2)*soluto*soluto)/TAU_R
# r = (Scalar(1) - (1/B**2)*soluto*soluto)/TAU_R

lhs_soluto = massa.mass/DT*soluto + div*upwind.upwind*soluto - r
rhs_soluto = massa.mass/DT*soluto_0 + div*upwind.rhs*bound_ad
eqn_soluto = pp.ad.Expression(lhs_soluto - rhs_soluto, dof, name='eqn_soluto')

lhs_precipitato = massa.mass/DT*precipitato + r
rhs_precipitato = massa.mass/DT*precipitato_0
eqn_precipitato = pp.ad.Expression(lhs_precipitato - rhs_precipitato, dof, name='eqn_precipitato')

equation_manager.equations += [eqn_soluto, eqn_precipitato]


''' SOLUZIONE '''

# TODO: Forse qui dovrei far convergere la soluzione di modo che
# le condizioni iniziali siano compatibili con le equazioni

d[pp.STATE]['soluto'][:] = d[pp.STATE]['soluto_0']
d[pp.STATE]['precipitato'][:] = d[pp.STATE]['precipitato_0']
exporter.write_vtu(['soluto', 'precipitato'], time_step=0)

for i in np.arange(1, int(T/DT)):
    eqn_soluto.discretize(gb)
    eqn_precipitato.discretize(gb)
    jacobiano, nresiduale = equation_manager.assemble_matrix_rhs()

    converged = False
    for k in np.arange(10):
        norma_residuale = np.max(np.abs(nresiduale))
        print(f'({i:2d},{k}): residuale: {norma_residuale:8.6f}')

        if norma_residuale < 1e-4:
            converged = True
            break

        incremento = sps.spsolve(jacobiano, nresiduale)
        dof.distribute_variable(incremento, additive=True)

        eqn_soluto.discretize(gb)
        eqn_precipitato.discretize(gb)
        jacobiano, nresiduale = equation_manager.assemble_matrix_rhs()

    if not converged:
        print('😓')
        break

    exporter.write_vtu(['soluto', 'precipitato'], time_step=i)
    d[pp.STATE]['soluto_0'][:] = d[pp.STATE]['soluto']
    d[pp.STATE]['precipitato_0'][:] = d[pp.STATE]['precipitato']

'''
Esempio erosione in https://doi.org/10.1016/j.jmaa.2015.06.003
Rotto
'''

import numpy as np
import porepy as pp
from porepy.numerics.ad.operators import Array
from porepy.numerics.fv.generaltpfaad import GeneralTpfaAd
import scipy.sparse as sps
import matplotlib.pyplot as plt
from helpers.upwind import Upwind
from helpers.trasmissibilita import trasmissibilita_da_permeabilita_
from helpers.vari import clamp, p_
from helpers.tpfa import Tpfa

X = 1
Y = 1
gb = pp.meshing.cart_grid([], [30, 30], physdims=[X, Y])
exporter = pp.Exporter(gb, file_name='soluzione', folder_name='out/2709u/')
g, d = [(g, d) for g, d in gb][0]

''' DATI, CONDIZIONI AL BORDO, CONDIZIONI INIZIALI '''
d[pp.PRIMARY_VARIABLES] = {
    'pressione': {'cells': 1},
    'soluto': {'cells': 1},
    'precipitato': {'cells': 1},
}
d[pp.STATE] = {}

T = 1.0
DT = 0.001

facce_bordo = g.tags["domain_boundary_faces"].nonzero()[0]
gamma1 = g.face_centers[0, facce_bordo] == 0
gamma2 = g.face_centers[1, facce_bordo] == 0
gamma3 = g.face_centers[0, facce_bordo] == X
gamma4 = g.face_centers[1, facce_bordo] == Y


''' darcy '''
valori_bc = np.zeros(g.num_faces)
tipi_bc = np.full(facce_bordo.size, 'nan')

tipi_bc[gamma1] = 'dir'; valori_bc[facce_bordo[gamma1]] = 0.5
tipi_bc[gamma2] = 'neu'; valori_bc[facce_bordo[gamma2]] = 0
tipi_bc[gamma3] = 'dir'; valori_bc[facce_bordo[gamma3]] = 0
tipi_bc[gamma4] = 'neu'; valori_bc[facce_bordo[gamma4]] = 0

bc = pp.BoundaryCondition(g, facce_bordo, tipi_bc)

# NOTE: nan per essere sicuro che non venga usato.
# La vera permeabilità la definisco più avanti
permeabilita = pp.SecondOrderTensor(np.full(g.num_cells, np.nan))
parametri_darcy = {"bc": bc, "bc_values": valori_bc, "second_order_tensor": permeabilita}
pp.initialize_default_data(g, d, 'flow', parametri_darcy)

d[pp.STATE]['pressione'] = np.zeros(g.num_cells)

''' soluto '''
valori_bc = np.zeros(g.num_faces)
tipi_bc = np.full(facce_bordo.size, 'nan')

tipi_bc[gamma1] = 'dir'; valori_bc[facce_bordo[gamma1]] = 0
tipi_bc[gamma2] = 'neu'; valori_bc[facce_bordo[gamma2]] = 0
tipi_bc[gamma3] = 'neu'; valori_bc[facce_bordo[gamma3]] = 0
tipi_bc[gamma4] = 'neu'; valori_bc[facce_bordo[gamma4]] = 0

bc = pp.BoundaryCondition(g, facce_bordo, tipi_bc)

# NOTE: Il vero flusso di Darcy lo definisco più avanti.
darcy = np.full(g.num_cells, np.nan)
diffusivita = pp.SecondOrderTensor(np.ones(g.num_cells))
parametri_trasporto = {"bc": bc, "bc_values": valori_bc, "mass_weight": np.ones(g.num_cells), "darcy_flux": darcy, "second_order_tensor": diffusivita}
pp.initialize_default_data(g, d, 'transport', parametri_trasporto)

d[pp.STATE]['soluto'] = np.zeros(g.num_cells)
d[pp.STATE]['soluto_0'] = np.zeros(g.num_cells)
d[pp.STATE]['soluto_0'][:] = 1 - 1e-6

''' precipitato e porosità '''

# NOTE: Nessuna condizione al bordo dato che non ho derivate spaziali nell'equazione del precipitato

d[pp.STATE]['precipitato'] = np.zeros(g.num_cells)
d[pp.STATE]['precipitato_0'] = np.zeros(g.num_cells)

d[pp.STATE]['precipitato_0'][:] = 1e-6
centro = (0.4 <= g.cell_centers[0, :]) & (g.cell_centers[0, :] <= 0.6) & (0.4 <= g.cell_centers[1, :]) & (g.cell_centers[1, :] <= 0.6)
d[pp.STATE]['precipitato_0'][centro] = 0.8

''' quantità derivate '''
d[pp.STATE]['_porosita'] = np.ones(g.num_cells)
d[pp.STATE]['_R'] = np.ones(g.num_cells)

''' EQUAZIONI '''

dof = pp.DofManager(gb); p = p_(gb, dof)
equation_manager = pp.ad.EquationManager(gb, dof)

pressione = equation_manager.merge_variables([(g, 'pressione')])
soluto = equation_manager.merge_variables([(g, 'soluto')])
precipitato = equation_manager.merge_variables([(g, 'precipitato')])

soluto_0 = Array(d[pp.STATE]['soluto_0'])
precipitato_0 = Array(d[pp.STATE]['precipitato_0'])

def porosita_da_precipitato(precipitato):
    porosita = 1 - precipitato # (A9)
    clamp(porosita)
    return porosita
porosita_da_precipitato_ad = pp.ad.Function(porosita_da_precipitato, 'porosita_da_precipitato')
porosita_0 = porosita_da_precipitato_ad(precipitato_0)


# NOTE: Questi due credo siano gli stessi, a prescindere da cosa sto discretizzando.
div = pp.ad.Divergence([g])
massa = pp.ad.MassMatrixAd('flow', [g])

porosita = porosita_da_precipitato_ad(precipitato)

''' darcy '''
ftrasmissibilita_da_permeabilita = trasmissibilita_da_permeabilita_(g)
ftrasmissibilita_da_permeabilita_ad = pp.ad.Function(ftrasmissibilita_da_permeabilita, 'ftrasmissibilita_da_permeabilita')
darcy_bc = pp.ad.BoundaryCondition('flow', grids=[g])

tpfa = Tpfa('flow', g, d)

permeabilita = porosita*porosita # (A9)
flusso = tpfa(ftrasmissibilita_da_permeabilita_ad(permeabilita), pressione, darcy_bc)

lhs_darcy = massa.mass/DT*porosita + div*flusso
rhs_darcy = massa.mass/DT*porosita_0
eqn_darcy = pp.ad.Expression(lhs_darcy - rhs_darcy, dof, name='eqn_darcy')

equation_manager.equations += [eqn_darcy]

''' soluto e precipitato '''
soluto_bc = pp.ad.BoundaryCondition('transport', grids=[g])

def R(u, v):
    d = 0.1

    # return u*0
    V = v/d; clamp(V)
    # V = 1/2*(pp.ad.tanh(v/d) + 1)

    # return 0*u
    return u*u - V
    # return (u*u - V) * (v > 1e-5)
    # return (u*u - 1) * (v > d)

R_ad = pp.ad.Function(R, 'R')

upwind = Upwind('transport', g, d)
tpfa = Tpfa('transport', g, d)

diffusivita = Array(np.ones(g.num_cells) * 1)

ttrasmissibilita_da_permeabilita = trasmissibilita_da_permeabilita_(g)
ttrasmissibilita_da_permeabilita_ad = pp.ad.Function(ttrasmissibilita_da_permeabilita, 'ttrasmissibilita_da_permeabilita')

# lhs_soluto = massa.mass/DT*soluto + R_ad(soluto, precipitato)
# lhs_soluto = massa.mass/DT*soluto + div*tpfa(ttrasmissibilita_da_permeabilita_ad(diffusivita), soluto, soluto_bc) + div*upwind(flusso, soluto, soluto_bc) + R_ad(soluto, precipitato)
# lhs_soluto = massa.mass/DT*soluto + div*tpfa(ttrasmissibilita_da_permeabilita_ad(diffusivita), soluto, soluto_bc) + R_ad(soluto, precipitato)
lhs_soluto = massa.mass/DT*soluto + div*upwind(flusso, soluto, soluto_bc) + R_ad(soluto, precipitato)
rhs_soluto = massa.mass/DT*soluto_0

eqn_soluto = pp.ad.Expression(lhs_soluto - rhs_soluto, dof, name='eqn_soluto')

lhs_precipitato = massa.mass/DT*precipitato
rhs_precipitato = massa.mass/DT*precipitato_0 + R_ad(soluto, precipitato)
eqn_precipitato = pp.ad.Expression(lhs_precipitato - rhs_precipitato, dof, name='eqn_precipitato')

equation_manager.equations += [eqn_soluto, eqn_precipitato]

''' SOLUZIONE '''

d[pp.STATE]['soluto'][:] = d[pp.STATE]['soluto_0']
d[pp.STATE]['precipitato'][:] = d[pp.STATE]['precipitato_0']

I = 0

def scrivi():
    d[pp.STATE]['_porosita'] = porosita_da_precipitato(d[pp.STATE]['precipitato'])
    d[pp.STATE]['_R'] = R(d[pp.STATE]['soluto'], d[pp.STATE]['precipitato'])

    exporter.write_vtu(['soluto', 'precipitato', 'pressione', '_porosita', '_R'], time_step=I)
scrivi(); I += 1

def avanza():
    global I

    equation_manager.discretize(gb)
    jacobiano, nresiduale = equation_manager.assemble_matrix_rhs()

    converged = False
    for k in np.arange(10):
        norma_residuale = np.max(np.abs(nresiduale))
        print(f'({I:3d}, {I*DT:.3f}, {k}): residuale: {norma_residuale:8.6f}')
        if norma_residuale < 1e-6:
            converged = True
            break

        incremento = sps.linalg.spsolve(jacobiano, nresiduale)
        if np.any(np.isnan(incremento)): break
        dof.distribute_variable(incremento, additive=True)

        equation_manager.discretize(gb)
        jacobiano, nresiduale = equation_manager.assemble_matrix_rhs()

    if not converged:
        print('😓')
        raise SystemError

    scrivi(); I += 1
    
    d[pp.STATE]['soluto_0'][:] = d[pp.STATE]['soluto']
    d[pp.STATE]['precipitato_0'][:] = d[pp.STATE]['precipitato']

while I < int(T/DT): avanza()

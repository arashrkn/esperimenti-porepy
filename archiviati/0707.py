'''
Colonna 1d.
Evolvo flusso, porosità, concentrazioni soluto e precipitato.
Equazioni 1,3,4,5 in https://arxiv.org/abs/2005.09437
'''

import numpy as np
import porepy as pp
from porepy.numerics.ad.operators import Scalar, Array
from porepy.numerics.fv.generaltpfaad import GeneralTpfaAd
import scipy.sparse as sps
import matplotlib
import matplotlib.pyplot as plt
from helpers.upwind import Upwind
from helpers.trasmissibilita import trasmissibilita_da_permeabilita_


X = 10
Y = 10
gb = pp.meshing.cart_grid([], [20, 20], physdims=[X, Y])
exporter = pp.Exporter(gb, file_name='soluzione', folder_name='out/0707/')
g, d = [(g, d) for g, d in gb][0]

''' DATI, CONDIZIONI AL BORDO, CONDIZIONI INIZIALI '''
d[pp.PRIMARY_VARIABLES] = {
    'pressione': {'cells': 1},
    'soluto': {'cells': 1},
    'precipitato': {'cells': 1},
}
d[pp.STATE] = {}

T = 40
DT = 0.5

TAU_R = 100
# MAX nel senso che per questi valori non ho più dissoluzione/precipitazione
SOLUTO_MAX = 0.05
PRECIPITATO_MAX = 1.0

PHI_INERTE = 0.9
ETA = 10
K0 = 1
PHI0 = 0.1

facce_bordo = g.tags["domain_boundary_faces"].nonzero()[0]

''' darcy '''
valori_bc = np.zeros(g.num_faces)
valori_bc[facce_bordo[g.face_centers[0, facce_bordo] == X]] = 2
# valori_bc[facce_bordo[g.face_centers[0, facce_bordo] == 0]] = 2

tipi_bc = np.full(facce_bordo.size, 'neu')
tipi_bc[g.face_centers[0, facce_bordo] == 0] = 'dir'
tipi_bc[g.face_centers[0, facce_bordo] == X] = 'dir'
bc = pp.BoundaryCondition(g, facce_bordo, tipi_bc)

# NOTE: nan per essere sicuro che non venga usato.
# La vera permeabilità la definisco più avanti
permeabilita = pp.SecondOrderTensor(np.full(g.num_cells, np.nan))
parametri_darcy = {"bc": bc, "bc_values": valori_bc, "second_order_tensor": permeabilita}
pp.initialize_default_data(g, d, 'flow', parametri_darcy)

d[pp.STATE]['pressione'] = np.zeros(g.num_cells)

''' soluto '''
valori_bc = np.zeros(g.num_faces)

tipi_bc = np.full(facce_bordo.size, 'dir')
tipi_bc[g.face_centers[1, facce_bordo] == 0] = 'neu'
tipi_bc[g.face_centers[1, facce_bordo] == Y] = 'neu'
bc = pp.BoundaryCondition(g, facce_bordo, tipi_bc)

# NOTE: Il vero flusso di Darcy lo definisco più avanti.
darcy = np.full(g.num_cells, np.nan)
parametri_trasporto = {"bc": bc, "bc_values": valori_bc, "mass_weight": np.ones(g.num_cells), "darcy_flux": darcy}
pp.initialize_default_data(g, d, 'transport', parametri_trasporto)

d[pp.STATE]['soluto'] = np.zeros(g.num_cells)
d[pp.STATE]['soluto_0'] = np.zeros(g.num_cells)
d[pp.STATE]['soluto_0'][:] = 0.0

''' precipitato e porosità '''
def porosita_da_precipitato(precipitato):
    porosita = (1 - PHI_INERTE)*(1 + ETA*precipitato)**-1
    return porosita
porosita_da_precipitato_ad = pp.ad.Function(porosita_da_precipitato, 'porosita_da_precipitato')

d[pp.STATE]['precipitato'] = np.zeros(g.num_cells)
d[pp.STATE]['precipitato_0'] = np.zeros(g.num_cells)

d[pp.STATE]['precipitato_0'][:] = 0.0
centro = (4 <= g.cell_centers[0, :]) & (g.cell_centers[0, :] <= 6) & (4 <= g.cell_centers[1, :]) & (g.cell_centers[1, :] <= 6)
d[pp.STATE]['precipitato_0'][centro] = 1.0
# d[pp.STATE]['precipitato_0'][:] = g.cell_centers[0,:] / 10 * 2
# d[pp.STATE]['precipitato_0'][g.cell_centers[0, :] <= 5] = 1
# d[pp.STATE]['precipitato_0'][:] = 1 - g.cell_centers[0,:]/X

''' quantità derivate '''
d[pp.STATE]['_porosita'] = np.ones(g.num_cells)
# d[pp.STATE]['rsoluto'] = np.ones(g.num_cells)
d[pp.STATE]['rprecipitato'] = np.ones(g.num_cells)
# d[pp.STATE]['equilibrio_soluto'] = np.ones(g.num_cells)

''' EQUAZIONI '''

dof = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof)

pressione = equation_manager.merge_variables([(g, 'pressione')])
soluto = equation_manager.merge_variables([(g, 'soluto')])
precipitato = equation_manager.merge_variables([(g, 'precipitato')])

soluto_0 = Array(d[pp.STATE]['soluto_0'])
precipitato_0 = Array(d[pp.STATE]['precipitato_0'])
porosita_0 = porosita_da_precipitato_ad(precipitato_0)

# NOTE: Questi due credo siano gli stessi, a prescindere da cosa sto discretizzando.
div = pp.ad.Divergence([g])
massa = pp.ad.MassMatrixAd('flow', [g])

porosita = porosita_da_precipitato_ad(precipitato)

''' darcy '''
trasmissibilita_da_permeabilita = trasmissibilita_da_permeabilita_(g)
trasmissibilita_da_permeabilita_ad = pp.ad.Function(trasmissibilita_da_permeabilita, 'trasmissibilita_da_permeabilita')
darcy_bc = pp.ad.BoundaryCondition('flow', grids=[g])

# NOTE: Non posso usare tpfa normale perchè AD non propagherebbe lo jacobiano della permeabilità.
tpfa = GeneralTpfaAd('flow')
tpfa.discretize(g, d)

permeabilita = Scalar(K0 / PHI0**2) * porosita*porosita
flusso = tpfa.flux(trasmissibilita_da_permeabilita_ad(permeabilita), pressione, darcy_bc)

lhs_darcy = div*flusso + massa.mass/DT*porosita
rhs_darcy = massa.mass/DT*porosita_0
eqn_darcy = pp.ad.Expression(lhs_darcy - rhs_darcy, dof, name='eqn_darcy')

equation_manager.equations += [eqn_darcy]

''' soluto e precipitato '''
soluto_bc = pp.ad.BoundaryCondition('transport', grids=[g])

r_diss = precipitato/PRECIPITATO_MAX * (Scalar(1) - (1/SOLUTO_MAX**2)*soluto*soluto)/TAU_R
r_prec = soluto/SOLUTO_MAX * (Scalar(1) - (1/PRECIPITATO_MAX**2)*precipitato*precipitato)/TAU_R
# r_prec = 0
r = r_diss - r_prec
# r = (Scalar(1) - (1/B**2)*soluto*soluto)/TAU_R
# r = 0

upwind = Upwind('transport', g, d)
lhs_soluto = massa.mass/DT*(porosita*soluto) + div*upwind(flusso, soluto, soluto_bc) - porosita*r
rhs_soluto = massa.mass/DT*(porosita_0*soluto_0)

eqn_soluto = pp.ad.Expression(lhs_soluto - rhs_soluto, dof, name='eqn_soluto')

lhs_precipitato = massa.mass/DT*(porosita*precipitato) + porosita*r
rhs_precipitato = massa.mass/DT*(porosita_0*precipitato_0)
eqn_precipitato = pp.ad.Expression(lhs_precipitato - rhs_precipitato, dof, name='eqn_precipitato')

equation_manager.equations += [eqn_soluto, eqn_precipitato]

''' SOLUZIONE '''

d[pp.STATE]['soluto'][:] = d[pp.STATE]['soluto_0']
d[pp.STATE]['precipitato'][:] = d[pp.STATE]['precipitato_0']

def scrivi(i):
    d[pp.STATE]['_porosita'] = porosita_da_precipitato(d[pp.STATE]['precipitato'])
    d[pp.STATE]['rsoluto'] = d[pp.STATE]['soluto'] * d[pp.STATE]['_porosita']
    d[pp.STATE]['rprecipitato'] = d[pp.STATE]['precipitato'] * d[pp.STATE]['_porosita']
    # d[pp.STATE]['equilibrio_soluto'] = B * d[pp.STATE]['porosita']
    exporter.write_vtu(['soluto', 'precipitato', 'pressione', '_porosita', 'rsoluto', 'rprecipitato'], time_step=i)
scrivi(0)

def p(ad):
    exp = pp.ad.Expression(ad, dof)
    res = exp.to_ad(gb)
    return res

for i in np.arange(1, int(T/DT)):
    equation_manager.discretize(gb)
    jacobiano, nresiduale = equation_manager.assemble_matrix_rhs()

    converged = False
    for k in np.arange(10):
        norma_residuale = np.max(np.abs(nresiduale))
        print(f'({i:2d},{k}): residuale: {norma_residuale:8.6f}')

        if norma_residuale < 1e-4:
            converged = True
            break

        incremento = sps.linalg.spsolve(jacobiano, nresiduale)
        if np.any(np.isnan(incremento)): break
        dof.distribute_variable(incremento, additive=True)

        equation_manager.discretize(gb)
        jacobiano, nresiduale = equation_manager.assemble_matrix_rhs()

    if not converged:
        print('😓')
        break

    if np.any(d[pp.STATE]['precipitato'] < -1e-3): raise SystemError

    scrivi(i)
    d[pp.STATE]['soluto_0'][:] = d[pp.STATE]['soluto']
    d[pp.STATE]['precipitato_0'][:] = d[pp.STATE]['precipitato']

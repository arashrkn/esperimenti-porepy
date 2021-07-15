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

X = 10
Y = 1
gb = pp.meshing.cart_grid([], [120, 1], physdims=[X, Y])
exporter = pp.Exporter(gb, file_name='soluzione', folder_name='out/0707/')
g, d = [(g, d) for g, d in gb][0]

''' DATI, CONDIZIONI AL BORDO, CONDIZIONI INIZIALI '''
d[pp.PRIMARY_VARIABLES] = {
    'pressione': {'cells': 1},
    'soluto': {'cells': 1},
    'precipitato': {'cells': 1},
    'porosita': {'cells': 1},
}
d[pp.STATE] = {}

# T = 30
T = 2
DT = 0.5

TAU_R = 10
B = 0.8

# ETA = 0.01
ETA = 0.8
K0 = 1
PHI0 = 1

facce_bordo = g.tags["domain_boundary_faces"].nonzero()[0]

''' darcy '''
valori_bc = np.zeros(g.num_faces)
valori_bc[facce_bordo[g.face_centers[0, facce_bordo] == 0]] = 2

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
d[pp.STATE]['pressione_0'] = np.zeros(g.num_cells)

''' soluto '''
valori_bc = np.zeros(g.num_faces)

tipi_bc = np.full(facce_bordo.size, 'dir')
tipi_bc[g.face_centers[1, facce_bordo] == 0] = 'neu'
tipi_bc[g.face_centers[1, facce_bordo] == Y] = 'neu'
bc = pp.BoundaryCondition(g, facce_bordo, tipi_bc)

porosita = np.ones(g.num_cells)

# NOTE: Il vero flusso di Darcy lo definisco più avanti.
darcy = np.full(g.num_cells, np.nan)
parametri_trasporto = {"bc": bc, "bc_values": valori_bc, "mass_weight": porosita, "darcy_flux": darcy}
pp.initialize_default_data(g, d, 'transport', parametri_trasporto)

d[pp.STATE]['soluto'] = np.zeros(g.num_cells)
d[pp.STATE]['soluto_0'] = np.zeros(g.num_cells)
d[pp.STATE]['soluto_0'][:] = 0

''' precipitato '''
d[pp.STATE]['precipitato'] = np.zeros(g.num_cells)
d[pp.STATE]['precipitato_0'] = np.zeros(g.num_cells)
d[pp.STATE]['precipitato_0'][g.cell_centers[0, :] <= 5] = 1

''' porosita '''
d[pp.STATE]['porosita'] = np.zeros(g.num_cells)
d[pp.STATE]['porosita_0'] = np.zeros(g.num_cells)
d[pp.STATE]['porosita_0'][:] = 0.1


''' EQUAZIONI '''

dof = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof)

pressione = equation_manager.merge_variables([(g, 'pressione')])
soluto = equation_manager.merge_variables([(g, 'soluto')])
precipitato = equation_manager.merge_variables([(g, 'precipitato')])
porosita = equation_manager.merge_variables([(g, 'porosita')])

pressione_0 = Array(d[pp.STATE]['pressione_0'])
soluto_0 = Array(d[pp.STATE]['soluto_0'])
precipitato_0 = Array(d[pp.STATE]['precipitato_0'])
porosita_0 = Array(d[pp.STATE]['porosita_0'])

# NOTE: Questi due credo siano gli stessi, a prescindere da cosa sto discretizzando.
div = pp.ad.Divergence([g])
massa = pp.ad.MassMatrixAd('flow', [g])

''' darcy '''

def trasmissibilita_da_porosita_(g):
    facce_segnate = np.vstack(g.cell_faces.nonzero())
    J = facce_segnate.shape[1]
    A = sps.dok_matrix((J, g.num_cells), dtype=np.float32)
    for j in range(J):
        idx_faccia = facce_segnate[0, j]
        idx_cella = facce_segnate[1, j]
        raggio = g.cell_centers[:, idx_cella] - g.face_centers[:, idx_faccia]
        A[j, idx_cella] = g.face_areas[idx_faccia] * np.dot(raggio, g.face_normals[:, idx_faccia]) / np.linalg.norm(raggio)
    A = np.abs(A)

    righe = facce_segnate[0, :]
    colonne = np.arange(J)
    dati = np.ones(J)
    B = sps.coo_matrix((dati, (righe, colonne)))
    B = B / np.array(B.sum(axis=1))
    B = sps.coo_matrix(B)

    def trasmissibilita_da_porosita(porosita):
        permeabilita = K0 / PHI0**2 * porosita**2
        inv_mezze_trasmissibilita = (A * permeabilita)**-1
        inv_trasmissibilita = B * inv_mezze_trasmissibilita
        trasmissibilita_ = inv_trasmissibilita**-1
        return trasmissibilita_

    return trasmissibilita_da_porosita
trasmissibilita_da_porosita_ad = pp.ad.Function(trasmissibilita_da_porosita_(g), 'trasmissibilita_da_porosita')

darcy_bc = pp.ad.BoundaryCondition('flow', grids=[g])

# NOTE: Non posso usare tpfa normale perchè AD non propagherebbe lo jacobiano della permeabilità.
tpfa = GeneralTpfaAd('flow')
tpfa.discretize(g, d)

flusso = tpfa.flux(trasmissibilita_da_porosita_ad(porosita), pressione, darcy_bc)

lhs_darcy = div * flusso + massa.mass/DT * porosita
rhs_darcy = massa.mass/DT*porosita_0
eqn_darcy = pp.ad.Expression(lhs_darcy - rhs_darcy, dof, name='eqn_darcy')

equation_manager.equations += [eqn_darcy]

''' soluto e precipitato '''
soluto_bc = pp.ad.BoundaryCondition('transport', grids=[g])

r = 0.5*precipitato * (Scalar(1) - (1/B**2)*soluto*soluto)/TAU_R
# r = (Scalar(1) - (1/B**2)*soluto*soluto)/TAU_R

# # NOTE: Non DEVO usare upwind così: AD non sta propagando lo jacobiano del flusso.
# upwind = pp.ad.UpwindAd('transport', [g])
# lhs_soluto = massa.mass/DT*(porosita*soluto) + div*upwind.upwind*soluto - porosita*r
# rhs_soluto = massa.mass/DT*(porosita_0*soluto_0) + div*upwind.rhs*soluto_bc

class Upwind(pp.ad.operators.ApplicableOperator):
    def __init__(self, keyword, g, data):
        self.keyword = keyword
        self.g = g
        self.data = data
        self._set_tree()

    def apply(self, flusso, concentrazioni, valori_bc):
        keyword = self.keyword
        g = self.g
        data = self.data
        bc = data[pp.PARAMETERS][keyword]['bc']

        div = g.cell_faces.T

        flusso_val = flusso.val if isinstance(flusso, pp.ad.Ad_array) else flusso
        flussi = np.einsum('ij,j->ij', div.toarray(), flusso_val)
        upwind = sps.csr_matrix(1*(flussi > 0)).T

        # NOTE: Su Neumann outflow ci pensa già upwind

        facce_neumann_inflow = np.logical_and(np.any(flussi < 0, axis=0), bc.is_neu).nonzero()[0]
        if facce_neumann_inflow.size > 0:
            raise SystemError()

        concentrazioni_facce = upwind * concentrazioni
        concentrazioni_facce.val[bc.is_dir] = valori_bc[bc.is_dir]
        concentrazioni_facce.jac[bc.is_dir, :] = 0
        concentrazioni_facce.jac.eliminate_zeros()

        res = concentrazioni_facce * flusso

        return res

upwind = Upwind('transport', g, d)
lhs_soluto = massa.mass/DT*(porosita*soluto) + div*upwind(flusso, soluto, soluto_bc) - porosita*r
rhs_soluto = massa.mass/DT*(porosita_0*soluto_0)

eqn_soluto = pp.ad.Expression(lhs_soluto - rhs_soluto, dof, name='eqn_soluto')

lhs_precipitato = massa.mass/DT*(porosita*precipitato) + porosita*r
rhs_precipitato = massa.mass/DT*(porosita_0*precipitato_0)
eqn_precipitato = pp.ad.Expression(lhs_precipitato - rhs_precipitato, dof, name='eqn_precipitato')

equation_manager.equations += [eqn_soluto, eqn_precipitato]

''' porosita '''
log_ad = pp.ad.Function(pp.ad.log, 'log')
lhs_porosita = log_ad(porosita) + ETA*precipitato
rhs_porosita = log_ad(porosita_0) + ETA*precipitato_0
# lhs_porosita =  ETA*massa.mass/DT*(porosita*precipitato) - ETA*massa.mass/DT*(porosita*precipitato_0) + massa.mass/DT*porosita
# rhs_porosita =  massa.mass/DT*porosita_0
eqn_porosita = pp.ad.Expression(lhs_porosita - rhs_porosita, dof, name='eqn_porosita')

equation_manager.equations += [eqn_porosita]


''' SOLUZIONE '''

d[pp.STATE]['pressione'] = d[pp.STATE]['pressione_0']
d[pp.STATE]['soluto'][:] = d[pp.STATE]['soluto_0']
d[pp.STATE]['precipitato'][:] = d[pp.STATE]['precipitato_0']
d[pp.STATE]['porosita'][:] = d[pp.STATE]['porosita_0']
exporter.write_vtu(['soluto', 'precipitato', 'pressione', 'porosita'], time_step=0)

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
        dof.distribute_variable(incremento, additive=True)

        equation_manager.discretize(gb)
        jacobiano, nresiduale = equation_manager.assemble_matrix_rhs()

    if not converged:
        print('😓')
        break

    exporter.write_vtu(['soluto', 'precipitato', 'pressione', 'porosita'], time_step=i)
    d[pp.STATE]['soluto_0'][:] = d[pp.STATE]['soluto']
    d[pp.STATE]['precipitato_0'][:] = d[pp.STATE]['precipitato']
    d[pp.STATE]['pressione_0'][:] = d[pp.STATE]['pressione']
    d[pp.STATE]['porosita_0'][:] = d[pp.STATE]['porosita']

'''
Upwind
'''

import numpy as np
import porepy as pp
from porepy.numerics.ad.operators import Scalar, Array
import scipy.sparse as sps
from .vari import azzera_riga


# TODO: Leeento!
# TODO: Come si estende al caso con più di una griglia?
# Si concatenano le matrici una dietro l'altra con sps.block_diag
# Vedi riga 463 in porepy/src/porepy/numerics/ad/discretizations.py
class Upwind(pp.ad.operators.ApplicableOperator):
    def __init__(self, keyword, g, data):
        self.keyword = keyword
        self.g = g
        self.data = data
        self.gialamentato = False
        self._set_tree()

    def apply(self, flusso, concentrazioni, valori_bc):
        keyword = self.keyword
        g = self.g
        data = self.data
        bc = data[pp.PARAMETERS][keyword]['bc']

        div = g.cell_faces.T

        # NOTE: Qui posso anche evitare di guardare lo jacobiano dato che 
        # una piccola variazione del flusso non mi fa cambiare direzione di upwind
        flusso_val = flusso.val if isinstance(flusso, pp.ad.Ad_array) else flusso
        
        flussi_dir = div.multiply(flusso_val); flussi_dir.data = np.sign(flussi_dir.data)
        upwind = (1 * (flussi_dir > 0)).transpose()
        facce_inflow = np.array(flussi_dir.sum(axis=0) == -1)[0]
        facce_dirichlet_inflow = facce_inflow & bc.is_dir
        facce_neumann_inflow = facce_inflow & bc.is_neu

        if np.any(facce_neumann_inflow) and not(self.gialamentato):
            print('OCCHIO: Inflow su bordo Neumann. Faccio finta di avere Dirichlet = 0')
            self.gialamentato = True    

        concentrazioni_facce = upwind * concentrazioni

        concentrazioni_facce.val[facce_dirichlet_inflow] = valori_bc[facce_dirichlet_inflow] # su Dirichlet outflow guardo al valore dentro al dominio, non devo guardare alla condizione al bordo
        concentrazioni_facce.val[facce_neumann_inflow] = 0
        
        concentrazioni_facce.jac = concentrazioni_facce.jac.tocsr()
        for riga in facce_inflow.nonzero()[0]: azzera_riga(concentrazioni_facce.jac, riga)
        concentrazioni_facce.jac.eliminate_zeros()
        concentrazioni_facce.jac = concentrazioni_facce.jac.tocsc()

        res = concentrazioni_facce * flusso

        return res

# TODO: Potrei metterci un piccolo test qui dentro.
# Da copiare da 0707.py che sta già usando la classe sopra.
def run():
    X = 30
    Y = 30
    gb = pp.meshing.cart_grid([], [X, Y], physdims=[X, Y])
    g, d = [(g, d) for g, d in gb][0]

    ''' DATI, CONDIZIONI AL BORDO, CONDIZIONI INIZIALI '''
    d[pp.PRIMARY_VARIABLES] = { 'soluto': {'cells': 1} }
    d[pp.STATE] = {}

    T = 5
    DT = 1

    facce_bordo = g.tags["domain_boundary_faces"].nonzero()[0]
    facce_bordo

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

    d[pp.STATE]['flusso'] = np.random.rand(g.num_faces) - 0.5
    flusso = Array(d[pp.STATE]['flusso'])

    ''' EQUAZIONI '''

    dof = pp.DofManager(gb)
    equation_manager = pp.ad.EquationManager(gb, dof)

    soluto = equation_manager.merge_variables([(g, 'soluto')])
    soluto_0 = Array(d[pp.STATE]['soluto_0'])

    div = pp.ad.Divergence([g])
    massa = pp.ad.MassMatrixAd('transport', [g])

    ''' soluto e precipitato '''
    soluto_bc = pp.ad.BoundaryCondition('transport', grids=[g])
    upwind = Upwind('transport', g, d)
    lhs_soluto = massa.mass/DT*(soluto) + div*upwind(flusso, soluto, soluto_bc)
    rhs_soluto = massa.mass/DT*(soluto_0)

    eqn_soluto = pp.ad.Expression(lhs_soluto - rhs_soluto, dof, name='eqn_soluto')
    equation_manager.equations = [eqn_soluto]

    ''' SOLUZIONE '''

    d[pp.STATE]['soluto'][:] = d[pp.STATE]['soluto_0']

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
            break

        d[pp.STATE]['soluto_0'][:] = d[pp.STATE]['soluto']


if __name__ == "__main__": run()




# NOTE: Bozza su come trattare Neumann inflow

# facce_neumann_inflow = np.logical_and(np.any(flussi < 0, axis=0), bc.is_neu).nonzero()[0]
# celle_neumann_inflow = g.cell_faces[facce_neumann_inflow, :].nonzero()[1]
# gradienti_di_concentrazione = valori_bc[facce_neumann_inflow]*g.face_normals[:, facce_neumann_inflow]*np.array(div[celle_neumann_inflow, facce_neumann_inflow])[0]
# delta_x = 2*(g.face_centers[:, facce_neumann_inflow] - g.cell_centers[:, celle_neumann_inflow])
# delta_c = np.zeros(g.num_cells)
# delta_c[celle_neumann_inflow] = np.sum(gradienti_di_concentrazione*delta_x, axis=0)
# solo_neumann_inflow = sps.coo_matrix((np.ones(celle_neumann_inflow.size), (celle_neumann_inflow, celle_neumann_inflow)), shape=(g.num_cells, g.num_cells))
# da_celle_a_facce = sps.coo_matrix((np.ones(celle_neumann_inflow.size), (facce_neumann_inflow, celle_neumann_inflow)), shape=(g.num_faces, g.num_cells))
# c_fuori = solo_neumann_inflow * (concentrazioni + delta_c)

# concentrazioni_facce = upwind * concentrazioni + da_celle_a_facce*c_fuori

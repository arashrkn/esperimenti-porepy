'''
Come 1906_2.py; in più flusso dato da soluzione iniziale di Darcy.
'''

import numpy as np
import porepy as pp
from porepy.numerics.ad.operators import Scalar, Array
import scipy.sparse as sps
import matplotlib.pyplot as plt

X = 10
Y = 1
gb = pp.meshing.cart_grid([], [10, 1], physdims=[X, Y])
exporter = pp.Exporter(gb, file_name='soluzione', folder_name='out/3006/')
g, d = [(g, d) for g, d in gb][0]

''' DATI, CONDIZIONI AL BORDO, CONDIZIONI INIZIALI '''
d[pp.PRIMARY_VARIABLES] = {'pressione': {'cells': 1}, 'soluto': {'cells': 1}, 'precipitato': {'cells': 1}}
d[pp.STATE] = {}

TAU_R = 10
B = 0.8
DT = 0.5
T = 75

facce_bordo = g.tags["domain_boundary_faces"].nonzero()[0]

''' darcy '''
valori_bc = np.zeros(g.num_faces)
valori_bc[facce_bordo[g.face_centers[0, facce_bordo] == 0]] = 2

tipi_bc = np.full(facce_bordo.size, 'neu')
tipi_bc[g.face_centers[0, facce_bordo] == 0] = 'dir'
tipi_bc[g.face_centers[0, facce_bordo] == X] = 'dir'
bc = pp.BoundaryCondition(g, facce_bordo, tipi_bc)

permeabilita = pp.SecondOrderTensor(np.ones(g.num_cells))

parametri_darcy = {"second_order_tensor": permeabilita, "bc": bc, "bc_values": valori_bc}
pp.initialize_default_data(g, d, 'flow', parametri_darcy)

d[pp.STATE]['pressione'] = np.zeros(g.num_cells)

''' soluto '''
valori_bc = np.zeros(g.num_faces)

tipi_bc = np.full(facce_bordo.size, 'dir')
tipi_bc[g.face_centers[1, facce_bordo] == 0] = 'neu'
tipi_bc[g.face_centers[1, facce_bordo] == Y] = 'neu'
bc = pp.BoundaryCondition(g, facce_bordo, tipi_bc)

porosita = np.ones(g.num_cells)
darcy = np.zeros(g.num_faces)

parametri_trasporto = {"bc": bc, "bc_values": valori_bc, "mass_weight": porosita, "darcy_flux": darcy }
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

''' darcy '''
pressione = equation_manager.merge_variables([(g, 'pressione')])

div = pp.ad.Divergence([g])
bound_ad = pp.ad.BoundaryCondition('flow', grids=[g])
mpfa = pp.ad.MpfaAd('flow', [g])

lhs_darcy = div * mpfa.flux * pressione
rhs_darcy = -1 * div * mpfa.bound_flux * bound_ad
eqn_darcy = pp.ad.Expression(lhs_darcy - rhs_darcy, dof, name='eqn_darcy')

equation_manager.equations += [eqn_darcy]

''' soluto '''
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

''' darcy '''
eqn_darcy.discretize(gb)
jacobiano, nresiduale = equation_manager.assemble_matrix_rhs(equations=['eqn_darcy'], ad_var=[pressione])

incremento = sps.linalg.spsolve(jacobiano, nresiduale)
dof.distribute_restricted_variable(incremento, additive=True, variable_names=['pressione'])

pp.fvutils.compute_darcy_flux(gb, p_name='pressione', keyword_store='transport')

''' trasporto '''
d[pp.STATE]['soluto'][:] = d[pp.STATE]['soluto_0']
d[pp.STATE]['precipitato'][:] = d[pp.STATE]['precipitato_0']
exporter.write_vtu(['soluto', 'precipitato'], time_step=0)

for i in np.arange(1, int(T/DT)):
    eqn_soluto.discretize(gb)
    eqn_precipitato.discretize(gb)
    jacobiano, nresiduale = equation_manager.assemble_matrix_rhs(equations=['eqn_soluto', 'eqn_precipitato'], ad_var=[soluto, precipitato])

    converged = False
    for k in np.arange(10):
        norma_residuale = np.max(np.abs(nresiduale))
        print(f'({i:2d},{k}): residuale: {norma_residuale:8.6f}')

        if norma_residuale < 1e-4:
            converged = True
            break

        incremento = sps.linalg.spsolve(jacobiano, nresiduale)
        dof.distribute_restricted_variable(incremento, additive=True, variable_names=['soluto', 'precipitato'])

        eqn_soluto.discretize(gb)
        eqn_precipitato.discretize(gb)
        jacobiano, nresiduale = equation_manager.assemble_matrix_rhs(equations=['eqn_soluto', 'eqn_precipitato'], ad_var=[soluto, precipitato])

    if not converged:
        print('😓')
        break

    exporter.write_vtu(['soluto', 'precipitato'], time_step=i)
    d[pp.STATE]['soluto_0'][:] = d[pp.STATE]['soluto']
    d[pp.STATE]['precipitato_0'][:] = d[pp.STATE]['precipitato']



''' dovuto aggiungere in dof_manager.py  '''
raise SystemExit
def distribute_restricted_variable(
    self,
    values: np.ndarray,
    variable_names: Optional[List[str]] = None,
    iterate: bool = False,
    additive: bool = False,
) -> None:
    """Distribute a vector to the nodes and edges in the GridBucket.
    The intended use is to split a multi-physics solution vector into its
    component parts.
    Parameters:
        values (np.array): Vector to be split. It is assumed that it corresponds
            to the ordering implied in block_dof and full_dof, e.g. that it is
            the solution of a linear system assembled with the assembler.
        variable_names (list of str, optional): Names of the variable to be
            distributed. If not provided, all variables found in block_dof
            will be distributed
        additive (bool, optional): If True, the variables are added to the current
            state, instead of overwrite the existing value.
    """
    if variable_names is None:
        variable_names = []
        for pair in self.block_dof.keys():
            variable_names.append(pair[1])

    # Make variable_names unique
    ''' variable_names = list(set(variable_names)) '''
    variable_names = variable_names

    # Determine restricted dof set for provided variables
    var_num_dofs = [self.num_dofs(var=v) for v in variable_names]
    dof = np.cumsum(np.append(0, np.asarray(var_num_dofs)))

    for var_name in set(variable_names):
        vi = variable_names.index(var_name)
        for pair, bi in self.block_dof.items():
            g = pair[0]
            name = pair[1]
            if name != var_name:
                continue
            if isinstance(g, tuple):
                # This is really an edge
                data = self.gb.edge_props(g)
            else:
                data = self.gb.node_props(g)

            if not(iterate):
                if pp.STATE in data.keys():
                    vals = values[dof[vi] : dof[vi + 1]]
                    ''' print(f'Pulling {name} from ({dof[vi]},{dof[vi+1]})') '''
                    if additive:
                        vals += data[pp.STATE][var_name]

                    data[pp.STATE][var_name] = vals
                else:
                    # If no values exist, there is nothing to add to
                    data[pp.STATE] = {var_name: values[dof[vi] : dof[vi + 1]]}
            else:
                if pp.STATE in data.keys():
                    vals = values[dof[vi] : dof[vi + 1]]
                    if pp.ITERATE in data[pp.STATE].keys():
                        if additive:
                            vals += data[pp.STATE][pp.ITERATE][var_name]

                        data[pp.STATE][pp.ITERATE][var_name] = vals
                    else:
                        data[pp.STATE][pp.ITERATE] = {}
                        data[pp.STATE][pp.ITERATE][var_name] = vals

                else:
                    # If no values exist, there is nothing to add to
                    data[pp.STATE] = {}
                    data[pp.STATE][pp.ITERATE] = {var_name: values[dof[vi] : dof[vi + 1]]}

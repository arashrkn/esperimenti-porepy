'''
Darcy con two-point su mezzo fratturato
'''

import numpy as np
import porepy as pp
import scipy.sparse as sps

class TP(pp.ad.Discretization):
    class TP_:
        def __init__(self, keyword):
            self.keyword = keyword

            self.diff_matrix_key = 'diff'
            self.dirichlet_matrix_key = 'dirichlet'
            self.neumann_matrix_key = 'neumann'

            self.mt_da_permeabilita_matrix_key = 'mt_da_permeabilita'
            self.somma_mf_matrix_key = 'somma_mf'
            self.azzera_neumann_matrix_key = 'azzera_neumann'

            self.traccia_p_dirichlet_matrix_key = 'traccia_p_dirichlet'
            self.traccia_p_neumann_matrix_key = 'traccia_p_neumann'
            self.traccia_p_correzione_neumann_matrix_key = 'traccia_p_correzione_neumann'

        def discretize(self, g, data):
            matrici = data[pp.DISCRETIZATION_MATRICES][self.keyword]
            parametri = data[pp.PARAMETERS][self.keyword]
            
            if g.dim == 0:
                nf = 0 # = g.num_faces
                nmf = 0 # = numero mezze facce

                matrici['diff'] = sps.csr_matrix((nf, g.num_cells))
                matrici['dirichlet'] = sps.csr_matrix((nf, nf))
                matrici['neumann'] = sps.csr_matrix((nf, nf))

                matrici['mt_da_permeabilita'] = sps.csr_matrix((nmf, g.num_cells))
                matrici['somma_mf'] = sps.csr_matrix((nf, nmf))
                matrici['azzera_neumann'] = sps.csr_matrix((nf, nf))

                matrici['traccia_p_dirichlet'] = sps.csr_matrix((nf, nf))
                matrici['traccia_p_neumann'] = sps.csr_matrix((nf, g.num_cells))
                matrici['traccia_p_correzione_neumann'] = sps.csr_matrix((nf, nmf))

                return

            bnd = parametri["bc"]
            bnd_mask = np.full(g.num_faces, False); bnd_mask[bnd.bf] = True

            # NOTE: Come si usano le matrici sotto
            # mezze_trasmissibilita = tp.mt_da_permeabilita * permeabilita
            # trasmissibilita = tp.azzera_neumann * inv_ad(tp.somma_mf * inv_ad(mezze_trasmissibilita)) 
            # flusso = (-1)*trasmissibilita*(tp.diff*pressione + tp.dirichlet*bound_ad) + tp.neumann*bound_ad
            # traccia_pressione = tp.traccia_p_dirichlet*bound_ad + tp.traccia_p_neumann*pressione + (tp.traccia_p_correzione_neumann*inv_ad(mezze_trasmissibilita))*(bound_ad + mortar_proj.mortar_to_primary_int*laambda)

            # inv(somma(inv( . ))) e' la media armonica(*0.5) (come per resistenze in parallelo)
            # azzero neumann dato che sui bordi neumann il flusso non e' dato dalla differenza di pressioni ma e' dettato dal valore della condizione al bordo
            # traccia_p_correzione_neumann deve moltiplicare dei flussi che sono uscenti
            # - bound_ad su neumann e' per definizione uscente
            # - i lambda invece hanno sempre il segno entrante nel mortar => uscente per la griglia che sta in alto

            # NOTE: Le matrici qui sotto dipendono solo da proprieta' geometriche e condizioni al contorno. Basta assemblarle una volta sola.

            segni_facce_bordo = np.array(g.cell_faces.sum(axis=1))[:,0] * bnd_mask

            div = g.cell_faces.T
            diff = -div.T
            dirichlet = sps.diags(segni_facce_bordo*bnd.is_dir)
            neumann = sps.diags(segni_facce_bordo*bnd.is_neu)

            matrici['diff'] = diff
            matrici['dirichlet'] = dirichlet
            matrici['neumann'] = neumann

            # mf: mezze facce
            facce_mf, celle_mf, segni_mf = sps.find(g.cell_faces)
            num_mf = facce_mf.size
            mf = np.arange(num_mf)

            # NOTE: le face_normals non sono normalizzate. hanno come lunghezza l'area della faccia
            normali_mf = g.face_normals[:,facce_mf] * segni_mf
            vettori_mf = g.face_centers[:,facce_mf] - g.cell_centers[:,celle_mf]
            distanza_mf_sq = np.sum(vettori_mf * vettori_mf, axis=0)
            mt = np.sum(normali_mf * vettori_mf, axis=0) / distanza_mf_sq
            # assert(np.allclose(mt*np.sqrt(distanza_mf_sq), g.face_areas[facce_mf])) # su griglie cartesiane
            mt_da_permeabilita = sps.coo_matrix(( mt, (mf, celle_mf) ))
            somma_mf = sps.csr_matrix(( np.ones(num_mf), (facce_mf, mf) ))
            azzera_neumann = sps.diags(1*np.logical_not(bnd.is_neu))

            matrici['mt_da_permeabilita'] = mt_da_permeabilita
            matrici['somma_mf'] = somma_mf
            matrici['azzera_neumann'] = azzera_neumann

            traccia_p_dirichlet = sps.diags(1*bnd.is_dir)
            traccia_p_neumann = sps.csr_matrix(( 1*bnd.is_neu[facce_mf], (facce_mf, celle_mf) ), shape=(g.num_faces, g.num_cells))
            traccia_p_correzione_neumann = sps.csr_matrix(( -1*bnd.is_neu[facce_mf], (facce_mf, mf) ), shape=(g.num_faces, num_mf))

            matrici['traccia_p_dirichlet'] = traccia_p_dirichlet
            matrici['traccia_p_neumann'] = traccia_p_neumann
            matrici['traccia_p_correzione_neumann'] = traccia_p_correzione_neumann
    
    def __init__(self, keyword, grids):
        self.grids = grids
        self._discretization = self.TP_(keyword)
        self._name = "TP"
        self.keyword = keyword

        pp.ad._ad_utils.wrap_discretization(self, self._discretization, grids=grids)

class MassaMortar(pp.ad.Discretization):
    class Massa_():
        def __init__(self, keyword):
            self.keyword = keyword
            self.massa_matrix_key = 'massa'

        def discretize(self, g_h, g_l, data_h, data_l, data_edge):
            matrici = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]

            # TODO: non sono del tutto sicuro che mi serva la massa nell'equazione dell'interfaccia
            # da fare chiarezza sulle dimensioni delle quantita': lambda e' m^3/s (volumetric flow) o m/s (flux) ?

            mg = data_edge["mortar_grid"]
            massa = sps.diags(mg.cell_volumes)
            matrici['massa'] = massa
    
    def __init__(self, keyword, edges):
        self.edges = edges
        self._discretization = self.Massa_(keyword)
        self._name = "Massa"
        self.keyword = keyword

        pp.ad._ad_utils.wrap_discretization(self, self._discretization, edges=edges)




if __name__ != '__main__': raise SystemExit

import matplotlib.pyplot as plt
from mesh import load_mesh

nome_mesh = 'C'
gb = load_mesh(nome_mesh)
exporter = pp.Exporter(gb, file_name='soluzione', folder_name='out/darcy/')

# # frattura *permeabile*
# permeabilita_matrice = 0.01
# permeabilita_fratture = 1.0
# # apertura_fratture = 0.010
# # apertura_fratture = 0.005   
# # apertura_fratture = 0.003
# apertura_fratture = 0.001

# frattura *impermeabile*
permeabilita_matrice = 1.0
permeabilita_fratture = 0.01
apertura_fratture = 0.001

for g, d in gb:
    d[pp.PRIMARY_VARIABLES] = { 'pressione': {'cells': 1}, }

    d[pp.STATE] = {}
    d[pp.STATE]['pressione'] = np.random.rand(g.num_cells)

for e, d in gb.edges():
    g = d['mortar_grid']
    d[pp.PRIMARY_VARIABLES] = {'laambda': {'cells': 1}}

    d[pp.STATE] = {}
    d[pp.STATE]['laambda'] = np.random.rand(g.num_cells)

gs = [g for g, _ in gb]
ms = [e for e, _ in gb.edges()]

ds = [d for _, d in gb]
mds = [d for _, d in gb.edges()]
mgs = [md['mortar_grid'] for md in mds]

for g,d in gb:
    if not g.dim in [2]: continue
    
    perm = pp.SecondOrderTensor(permeabilita_matrice * np.ones(g.num_cells))

    left, right = pp.face_on_side(g, 'west')[0], pp.face_on_side(g, 'east')[0]
    facce_dir = np.concatenate([ left, right ])
    bc = pp.BoundaryCondition(g, faces=facce_dir, cond=np.full(facce_dir.size, 'dir'))
    
    bc_val = np.zeros(g.num_faces)
    bc_val[left] = 2
    bc_val[right] = 0

    assert((bc.is_internal & ~bc.is_neu).sum() == 0)
    assert(np.all(bc_val[bc.is_internal] == 0))

    data = {'second_order_tensor': perm, 'bc': bc, 'bc_values': bc_val}
    d = pp.initialize_data(g, d, 'flow', data)

for g,d in gb:
    if not g.dim in [0,1]: continue

    if g.dim == 1: permeabilita_effettiva = permeabilita_fratture*apertura_fratture
    if g.dim == 0: permeabilita_effettiva = permeabilita_fratture*apertura_fratture*apertura_fratture
    perm = pp.SecondOrderTensor(permeabilita_effettiva * np.ones(g.num_cells))

    bc = pp.BoundaryCondition(g)
    bc_val = np.zeros(g.num_faces)

    data = {'second_order_tensor': perm, 'bc': bc, 'bc_values': bc_val}
    d = pp.initialize_data(g, d, 'flow', data)

for e, d in gb.edges():
    mg = d['mortar_grid']
    # TODO: giusto?
    if g.dim == 1: diffusivita_interfaccia = permeabilita_fratture/(apertura_fratture/2)
    if g.dim == 0: diffusivita_interfaccia = permeabilita_fratture*apertura_fratture/(apertura_fratture/2)
    kn = diffusivita_interfaccia * np.ones(mg.num_cells)
    pp.initialize_data(mg, d, 'flow', {'normal_diffusivity': kn})

dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)

pressione = equation_manager.merge_variables([(g, 'pressione') for g in gs])
laambda = equation_manager.merge_variables([(e, 'laambda') for e in ms])


num_grids = len(gs)
num_edges = len(ms)
grid_dof = np.cumsum(dof_manager.full_dof[0:num_grids]); grid_dof = np.insert(grid_dof, 0, 0)
edge_dof = np.cumsum(dof_manager.full_dof[num_grids:]); edge_dof = np.insert(edge_dof, 0, 0)

permeabilita = np.zeros(grid_dof[-1])
for i in range(0, num_grids): permeabilita[grid_dof[i]:grid_dof[i+1]][:] = ds[i][pp.PARAMETERS]['flow']['second_order_tensor'].values[0,0,:]

diffusivita = np.zeros(edge_dof[-1])
for i in range(0, num_edges): diffusivita[edge_dof[i]:edge_dof[i+1]][:] = mds[i][pp.PARAMETERS]['flow']['normal_diffusivity']

div = pp.ad.Divergence(gs)
mortar_proj = pp.ad.MortarProjections(gb=gb, grids=gs, edges=ms)
bound_ad = pp.ad.BoundaryCondition('flow', gs)
tp = TP('flow', gs)
massa_mortar = MassaMortar('flow', edges=ms)
def inv(x): return 1/x
inv_ad = pp.ad.Function(inv, 'inv')

mezze_trasmissibilita = tp.mt_da_permeabilita*permeabilita
trasmissibilita = tp.azzera_neumann*inv_ad( tp.somma_mf*inv_ad(mezze_trasmissibilita) )
flusso = (-1)*trasmissibilita*(tp.diff*pressione + tp.dirichlet*bound_ad) + tp.neumann*(bound_ad + mortar_proj.mortar_to_primary_int*laambda)
darcy = div*flusso + mortar_proj.mortar_to_secondary_int*laambda

traccia_pressione = tp.traccia_p_dirichlet*bound_ad + tp.traccia_p_neumann*pressione \
    + (tp.traccia_p_correzione_neumann*inv_ad(mezze_trasmissibilita))*(bound_ad + mortar_proj.mortar_to_primary_int*laambda)
pressione_sup = mortar_proj.primary_to_mortar_avg * traccia_pressione
pressione_inf = mortar_proj.secondary_to_mortar_avg * pressione
interfaccia = (massa_mortar.massa*diffusivita) * (pressione_inf - pressione_sup) + laambda

equation_manager.equations['darcy'] = darcy
equation_manager.equations['interfaccia'] = interfaccia

equation_manager.discretize(gb)

A, b = equation_manager.assemble()
soluzione = sps.linalg.spsolve(A, b)

dof_manager.distribute_variable(soluzione, additive=True)
exporter.write_vtu(['pressione']) # TODO: laambda non si puo' esportare?
# NOTE: (non so ancora dove) spiego perche' non uso compute_darcy_flux
darcy_flux = flusso.evaluate(dof_manager).val
np.savetxt(f"out/flussi/{nome_mesh}", darcy_flux)
# pp.plot_grid(gb, 'pressione')
raise SystemExit

tpfa = pp.ad.TpfaAd('flow', gs)
robin_ = pp.ad.RobinCouplingAd('flow', ms)

interior_flux = tpfa.flux * pressione
full_flux = interior_flux + tpfa.bound_flux * bound_ad + tpfa.bound_flux*mortar_proj.mortar_to_primary_int * laambda
sources_from_mortar = mortar_proj.mortar_to_secondary_int * laambda
conservation = div * full_flux + sources_from_mortar

pressure_trace_from_high = (
    mortar_proj.primary_to_mortar_avg * tpfa.bound_pressure_cell * pressione
    + mortar_proj.primary_to_mortar_avg * tpfa.bound_pressure_face * mortar_proj.mortar_to_primary_int * laambda
)
interface_flux_eq = robin_.mortar_discr * (pressure_trace_from_high - mortar_proj.secondary_to_mortar_avg * pressione) + laambda

eqs = {'subdomain_conservation': conservation, 'interface_fluxes': interface_flux_eq}
equation_manager.equations.clear()
equation_manager.equations.update(eqs)
equation_manager.discretize(gb)
A_, b_ = equation_manager.assemble()
soluzione_ = sps.linalg.spsolve(A_, b_)

def p_(t, name='...'): print(('  ' if t else 'no') + ' (' + name + ')')
t = np.allclose(soluzione, soluzione_); p_(t, 'Soluzione')
t = np.allclose(A.toarray(), A_.toarray()); p_(t, 'A')
t = np.allclose(b, b_); p_(t, 'b')

nuovo_ = (-1) * massa_mortar.massa * diffusivita
nuovo = sps.diags(nuovo_.evaluate(dof_manager))
mortar_discr = robin_.mortar_discr
classico = mortar_discr.evaluate(dof_manager)
t = np.allclose(nuovo.toarray(), classico.toarray()); p_(t, 'Mortar discretization')

nuovo_ = (pressione_sup - pressione_inf)
classico_ = (pressure_trace_from_high - mortar_proj.secondary_to_mortar_avg * pressione)
nuovo = nuovo_.evaluate(dof_manager)
classico = classico_.evaluate(dof_manager)
t = np.allclose(nuovo.val, classico.val); p_(t, 'Salto pressione (val)')
t = np.allclose(nuovo.jac.toarray(), classico.jac.toarray()); p_(t, 'Salto pressione (jac)')

flusso_interno = (-1)*trasmissibilita*(tp.diff*pressione)
flusso_interno.discretize(gb); nuovo = flusso_interno.evaluate(dof_manager)
interior_flux = tpfa.flux * pressione; interior_flux.discretize(gb); classico = interior_flux.evaluate(dof_manager)
t = np.allclose(nuovo.val, classico.val); p_(t, 'Flusso interno (val)')
t = np.allclose(nuovo.jac.toarray(), classico.jac.toarray()); p_(t, 'Flusso interno (jac)')

# np.savetxt('out/darcy/classico', np.vstack(sps.find(classico.jac)).T, fmt='%1.6f')
# np.savetxt('out/darcy/nuovo', np.vstack(sps.find(nuovo.jac)).T, fmt='%1.6f')

flusso_legato = (-1)*trasmissibilita*tp.dirichlet*bound_ad + tp.neumann*bound_ad
flusso_legato.discretize(gb); nuovo = flusso_legato.evaluate(dof_manager)
bound_flux = tpfa.bound_flux * bound_ad; bound_flux.discretize(gb); classico = bound_flux.evaluate(dof_manager)
t = np.allclose(nuovo, classico); p_(t, 'Flusso legato')

flusso_da_mortar = tp.neumann*(mortar_proj.mortar_to_primary_int*laambda)
flusso_da_mortar.discretize(gb); nuovo = flusso_da_mortar.evaluate(dof_manager)
mortar_flux = tpfa.bound_flux*mortar_proj.mortar_to_primary_int*laambda; mortar_flux.discretize(gb); classico = mortar_flux.evaluate(dof_manager)
t = np.allclose(nuovo.val, classico.val); p_(t, 'Flusso da mortar (val)')
t = np.allclose(nuovo.jac.toarray(), classico.jac.toarray()); p_(t, 'Flusso da mortar (jac)')

flusso.discretize(gb); nuovo = flusso.evaluate(dof_manager)
flux = tpfa.flux * pressione + tpfa.bound_flux * bound_ad + tpfa.bound_flux*mortar_proj.mortar_to_primary_int * laambda
flux.discretize(gb); classico = flux.evaluate(dof_manager)
t = np.allclose(nuovo.val, classico.val); p_(t, 'Flusso totale (val)')
t = np.allclose(nuovo.jac.toarray(), classico.jac.toarray()); p_(t, 'Flusso totale (jac)')

# np.savetxt('out/darcy/classico', np.vstack(sps.find(classico)).T, fmt='%1.6f')
# np.savetxt('out/darcy/nuovo', np.vstack(sps.find(nuovo)).T, fmt='%1.6f')

# data = ds[0]
# matrici = data[pp.DISCRETIZATION_MATRICES]['flow']
# parametri = data[pp.PARAMETERS]['flow']

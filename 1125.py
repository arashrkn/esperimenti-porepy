import numpy as np
import porepy as pp
import scipy.sparse as sps

frattura_1 = np.array([[0.2, 3.8], [0.5, 0.5]])
gb = pp.meshing.cart_grid( [frattura_1], nx=np.array([100, 40]), physdims=np.array([4,1]) )

nome_sim = 'viscous_t'
folder_name = f"out/{nome_sim}/"

import os
from shutil import copyfile
dest_path =  folder_name + os.path.basename(__file__)
os.makedirs(os.path.dirname(dest_path), exist_ok=True)
copyfile(__file__, dest_path)

exporter = pp.Exporter(gb, file_name=f"{nome_sim}", folder_name=folder_name)
rng = np.random.default_rng(seed=10)

U = 1
K_MATRICE = 1.0
K_FRATTURE = 20.0
APERTURA = 1e-4
R = 3
Pe = 512
D = U*1/Pe

for g,d in gb:
    x = g.cell_centers[0,:]
    d[pp.PRIMARY_VARIABLES] = { 'pressione': {'cells': 1}, 'soluto': {'cells': 1} }
    d[pp.STATE] = { 'pressione': np.ones(g.num_cells), 'soluto': np.ones(g.num_cells) }

for e,d in gb.edges():
    mg = d['mortar_grid']
    x = mg.cell_centers[0,:]

    d[pp.PRIMARY_VARIABLES] = { 'laambda_u': {'cells': 1}, 'laambda_c': {'cells': 1}, 'eta': {'cells': 1}  }
    d[pp.STATE] = { 'laambda_u': np.ones(mg.num_cells), 'laambda_c': np.ones(mg.num_cells), 'eta': np.ones(mg.num_cells) }

for g,d in gb:
    if not g.dim == 2: continue

    left, right = pp.face_on_side(g, 'west')[0], pp.face_on_side(g, 'east')[0]

    # TRASPORTO
    facce_dir = np.concatenate([ left, right ])
    bc = pp.BoundaryCondition(g, faces=facce_dir, cond=np.full(facce_dir.size, 'dir'))
    bc_val = np.zeros(g.num_faces)
    bc_val[left] = 1
    bc_val[right] = 0

    assert((bc.is_internal & ~bc.is_neu).sum() == 0)
    assert(np.all(bc_val[bc.is_internal] == 0))

    data = { 'bc': bc, 'bc_values': bc_val, 'darcy_flux': np.full(g.num_faces, np.nan), 'second_order_tensor': pp.SecondOrderTensor(np.ones(g.num_cells)), 'mass_weight': np.ones(g.num_cells) }
    pp.initialize_data(g, d, 'transport', data)

    # DARCY
    facce_dir = np.concatenate([ right ])
    bc = pp.BoundaryCondition(g, faces=facce_dir, cond=np.full(facce_dir.size, 'dir'))

    bc_val = np.zeros(g.num_faces)
    bc_val[left] = -U*g.face_areas[left]
    bc_val[right] = 0

    assert((bc.is_internal & ~bc.is_neu).sum() == 0)
    assert(np.all(bc_val[bc.is_internal] == 0))

    data = { 'bc': bc, 'bc_values': bc_val, 'second_order_tensor': pp.SecondOrderTensor(np.ones(g.num_cells)) }
    pp.initialize_data(g, d, 'flow', data)

for g,d in gb:
    if not g.dim in [0,1]: continue

    left = pp.face_on_side(g, 'west')[0]

    # TRASPORTO
    bc = pp.BoundaryCondition(g)
    bc_val = np.zeros(g.num_faces)

    # facce_dir = np.array([ left ])
    # bc = pp.BoundaryCondition(g, faces=facce_dir, cond=np.full(facce_dir.size, 'dir'))
    # bc_val = np.zeros(g.num_faces); bc_val[left] = 1

    data = {'darcy_flux': np.full(g.num_faces, np.nan), 'second_order_tensor': pp.SecondOrderTensor(np.ones(g.num_cells)), 'bc': bc, 'bc_values': bc_val, 'mass_weight': np.ones(g.num_cells) }
    pp.initialize_data(g, d, 'transport', data)

    # DARCY
    bc = pp.BoundaryCondition(g)
    bc_val = np.zeros(g.num_faces)

    # bc = pp.BoundaryCondition(g)
    # bc_val = np.zeros(g.num_faces); bc_val[left] = -U*g.face_areas[left]

    data = {'second_order_tensor': pp.SecondOrderTensor(np.ones(g.num_cells)), 'bc': bc, 'bc_values': bc_val}
    pp.initialize_data(g, d, 'flow', data)

for e,d in gb.edges():
    mg = d['mortar_grid']

    # TRASPORTO
    data = {'darcy_flux': np.full(mg.num_cells, np.nan), 'normal_diffusivity': np.ones(mg.num_cells)}
    pp.initialize_data(mg, d, 'transport', data)

    # DARCY
    data = {'normal_diffusivity': np.ones(mg.num_cells)}
    pp.initialize_data(mg, d, 'flow', data)


gs = [g for g, _ in gb]
es = [e for e, _ in gb.edges()]

ds = [d for _, d in gb]
mds = [d for _, d in gb.edges()]
mgs = [md['mortar_grid'] for md in mds]

dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)

pressione = equation_manager.merge_variables([(g, 'pressione') for g in gs])
soluto = equation_manager.merge_variables([(g, 'soluto') for g in gs])

div = pp.ad.Divergence(grids=gs)
mp = pp.ad.MortarProjections(gb=gb, grids=gs, edges=es)
trace = pp.ad.Trace(grids=gs)

def dir_(key, g, d):
    is_dir = d[pp.PARAMETERS][key]['bc'].is_dir
    bc_val = d[pp.PARAMETERS][key]['bc_values']
    dir = bc_val*is_dir
    return dir
def neu_(key, g, d):
    is_neu = d[pp.PARAMETERS][key]['bc'].is_neu
    bc_val = d[pp.PARAMETERS][key]['bc_values']
    neu = bc_val*is_neu
    return neu

flow = {
    'diff': pp.ad.MpfaAd('flow', gs), # [flux] = L^(d-2). [bound_flux] = 1 (neu), L^(d-2) (dir). [bound_pressure_face] =? L^(2-d)
    # 'bc': pp.ad.BoundaryCondition('flow', grids=gs),
    # 'rb_coupling': pp.ad.RobinCouplingAd('flow', es),
    'dir': pp.ad.Array( np.concatenate([ dir_('flow', g,d) for g,d in zip(gs, ds) ]), 'darcy_dir' ), # [] = Pa
    'neu': pp.ad.Array( np.concatenate([ neu_('flow', g,d) for g,d in zip(gs, ds) ]), 'darcy_neu' ), # [] = L^d/T
}

transport = {
    'mass': pp.ad.MassMatrixAd('transport', gs),
    'diff': pp.ad.MpfaAd('transport', gs), # [] = 1
    'upwind': pp.ad.UpwindAd('transport', gs),
    # 'bc': pp.ad.BoundaryCondition('transport', grids=gs),
    # 'rb_coupling': pp.ad.RobinCouplingAd('transport', es), # NOTE: se torno a riusare questo, da ricordare il segno nascosto dentro mortar_discr
    'up_coupling': pp.ad.UpwindCouplingAd('transport', es),
    'dir': pp.ad.Array( np.concatenate([ dir_('transport', g,d) for g,d in zip(gs, ds) ]), 'darcy_dir' ), # [] = C/L^2
    'neu': pp.ad.Array( np.concatenate([ neu_('transport', g,d) for g,d in zip(gs, ds) ]), 'darcy_neu' ), # TODO: [] = ??? 
}

def avg_(g):
    boundary = g.get_all_boundary_faces()
    weight_array = 0.5 * np.ones(g.num_faces)
    weight_array[boundary] = 1.0
    weights = sps.dia_matrix((weight_array, 0), shape=(g.num_faces, g.num_faces))
    media = weights * np.abs(g.cell_faces)
    return media
avg = sps.block_diag([avg_(g) for g in gs]).tocsr()
avg = pp.ad.Matrix(avg, 'avg')

def massa_mortar_(mg):
    volumes = mg.cell_volumes
    return volumes
massa_mortar = np.concatenate([ massa_mortar_(mg) for mg in mgs ])
massa_mortar = pp.ad.Array(massa_mortar) # [] = L^md

def inv_(x): return x**-1
def exp_(x): return pp.ad.exp(x)
def abs_(x): return pp.ad.abs(x)
inv = pp.ad.Function(inv_, 'inv')
exp = pp.ad.Function(exp_, 'exp')
abs = pp.ad.Function(abs_, 'abs')

dt = pp.ad.Scalar(np.nan)

pressione = equation_manager.merge_variables([(g, 'pressione') for g in gs]) # [] = Pa
laambda_u = equation_manager.merge_variables([(e, 'laambda_u') for e in es]) # [] = L^(md+1)/T
soluto = equation_manager.merge_variables([(g, 'soluto') for g in gs]) # [] = C/L^2
laambda_c = equation_manager.merge_variables([(e, 'laambda_c') for e in es]) # [] = CL^(md-1)/T
eta = equation_manager.merge_variables([(e, 'eta') for e in es]) # [] = CL^(md-1)/T

g_dims = np.concatenate([np.full(g.num_cells, g.dim) for g in gs])
mg_dims = np.concatenate([np.full(mg.num_cells, mg.dim) for mg in mgs])


soluto_0_ = np.zeros(gb.num_cells())
cell_dof = np.concatenate([ [0], np.cumsum([g.num_cells for g in gs]) ])
for i,g in enumerate(gs):
    inizio = g.cell_centers[0,:] < 0.15
    soluto_0_[cell_dof[i]:cell_dof[i+1]][inizio] = 1 - 0.05*rng.random(np.sum(inizio))
soluto_0 = pp.ad.Array(soluto_0_)

viscosita = 1*exp(-R*soluto) # [] = PaT
permeabilita = pp.ad.Array( np.select([g_dims == 0, g_dims == 1, g_dims == 2], [np.nan, K_FRATTURE, K_MATRICE]) ) # [] = L^2
diffusivita_darcy = permeabilita * inv(viscosita) # [] = L^2/PaT
diffusivita_darcy_l = avg*diffusivita_darcy

# [] = L^d/T
portata = (
      diffusivita_darcy_l * (flow['diff'].flux       * pressione)
    + diffusivita_darcy_l * (flow['diff'].bound_flux * flow['dir'])
    + flow['diff'].bound_flux * flow['neu']
    + flow['diff'].bound_flux * (mp.mortar_to_primary_int*laambda_u) # laambda_u entra tale e quale a neumann
)
# [] = L^2/T
darcy = APERTURA*(div*portata) - mp.mortar_to_secondary_int*laambda_u

# [] = Pa
traccia_pressione = (
      flow['diff'].bound_pressure_cell*pressione 
    + (flow['diff'].bound_pressure_face*mp.mortar_to_primary_int*laambda_u)/diffusivita_darcy_l
)
pressione_sup = mp.primary_to_mortar_avg*traccia_pressione
pressione_inf = mp.secondary_to_mortar_avg*pressione
diffusivita_darcy_int = mp.secondary_to_mortar_avg * (2*diffusivita_darcy/APERTURA) # [] = L/PaT
# [] = L^(md+1)/T
interfaccia_darcy = laambda_u + diffusivita_darcy_int*(massa_mortar*(pressione_inf - pressione_sup))

# [] = CL^(d-2)/T
portata_c_diff = (
      D * (transport['diff'].flux       * soluto)
    + D * (transport['diff'].bound_flux * transport['dir'])
    + transport['diff'].bound_flux * transport['neu']
    + transport['diff'].bound_flux * (mp.mortar_to_primary_int*laambda_c)
)
# NOTE: il secondo termine fa un po' schifo:
# non considero proprio neumann. anche perchè dalla teoria non so bene come va trattato
# NOTE: abs(portata) dato che in bound_transport per Dirichlet sono nascosti i segni del flusso (upwind.py:300)
# [] = CL^(d-2)/T
portata_c_avv = (
    portata*(transport['upwind'].upwind*soluto)
    - transport['upwind'].bound_transport*(abs(portata)*transport['dir'])
    - transport['upwind'].bound_transport*(mp.mortar_to_primary_int*eta)
)

# [] = CL^(d-1)/T
trasporto = (
      APERTURA*transport['mass'].mass*soluto/dt
    - APERTURA*transport['mass'].mass*soluto_0/dt
    + APERTURA*(div*portata_c_diff)
    + APERTURA*(div*portata_c_avv)
    - mp.mortar_to_secondary_int*laambda_c
    - mp.mortar_to_secondary_int*eta
)

soluto_inf = mp.secondary_to_mortar_avg*soluto
# TODO: nella correzione per la traccia non dev'esserci un contributo da parte della portata avvettiva?
traccia_soluto_diff = (
    transport['diff'].bound_pressure_cell*soluto 
    + (transport['diff'].bound_pressure_face*mp.mortar_to_primary_int*laambda_c)/D
)
soluto_sup_diff = mp.primary_to_mortar_avg*traccia_soluto_diff
# [] = CL^(md-1)/T
interfaccia_trasporto_diff = laambda_c + (2*D/APERTURA)*massa_mortar*(soluto_inf - soluto_sup_diff)

soluto_sup_upwind = mp.primary_to_mortar_avg*(trace.trace*soluto)
soluto_mortar = transport['up_coupling'].upwind_primary*soluto_sup_upwind + transport['up_coupling'].upwind_secondary*soluto_inf
# [] = CL^(md-1)/T
interfaccia_trasporto_avv = eta - laambda_u*soluto_mortar

equation_manager.equations['darcy'] = darcy
equation_manager.equations['interfaccia_darcy'] = interfaccia_darcy
equation_manager.equations['trasporto'] = trasporto
equation_manager.equations['interfaccia_trasporto_diff'] = interfaccia_trasporto_diff
equation_manager.equations['interfaccia_trasporto_avv'] = interfaccia_trasporto_avv

I = 0

def discretizza(primo_giro=False):
    if primo_giro: 
        darcy.discretize(gb)
        interfaccia_darcy.discretize(gb)
        trasporto.discretize(gb)
        interfaccia_trasporto_diff.discretize(gb)
        interfaccia_trasporto_avv.discretize(gb)

    # NOTE: devo mettere le portate nel dizionario dei parametri dato che upwind ne' guarda il segno
    facce_dof = np.concatenate([ [0], np.cumsum([g.num_faces for g in gs]) ])
    mcelle_dof = np.concatenate([ [0], np.cumsum([mg.num_cells for mg in mgs]) ])
    portata_val = portata.evaluate(dof_manager).val
    laambda_u_val = laambda_u.evaluate(dof_manager).val
    for i,d  in enumerate(ds): d[pp.PARAMETERS]['transport']['darcy_flux'][:] = portata_val[facce_dof[i]:facce_dof[i+1]]
    for i,md in enumerate(mds): md[pp.PARAMETERS]['transport']['darcy_flux'][:] = laambda_u_val[mcelle_dof[i]:mcelle_dof[i+1]]

    # NOTE: Le matrici in mass, mpfa dipendono solo da proprietà geometriche. Basta discretizzarle una volta sola
    transport['upwind'].upwind.discretize(gb)
    transport['up_coupling'].upwind_primary.discretize(gb)

def newton():
    converged = False
    for k in range(10):
        discretizza()
        A, b = equation_manager.assemble()

        norma_residuale = np.linalg.norm(b, np.inf)
        print(f'({I:3d}, {I*dt._value:.3f}, {k}): residuale: {norma_residuale:8.10f}')
        if norma_residuale < 1e-8:
            converged = True
            break

        incremento = sps.linalg.spsolve(A, b)
        if np.any(np.isnan(incremento)): break
        dof_manager.distribute_variable(incremento, additive=True)

    if not converged: raise SystemError


dt._value = 1e-8
discretizza(primo_giro=True)
newton()
exporter.write_vtu(['pressione', 'soluto'], time_step=I)
I += 1

dt._value = 0.05
for _ in range(50):
    newton()
    exporter.write_vtu(['pressione', 'soluto'], time_step=I)
    
    soluto_0._values = soluto.evaluate(dof_manager).val
    I += 1

# TODO: potrei scrivere sia
# diffusivita_darcy_n*(flow['rb_coupling'].mortar_discr*(pressione_sup - pressione_inf))
# che
# flow['rb_coupling'].mortar_discr*( diffusivita_darcy_n*(pressione_sup - pressione_inf) )
# il valore credo venga lo stesso (mortar_discr è diagonale)
# lo jacobiano?

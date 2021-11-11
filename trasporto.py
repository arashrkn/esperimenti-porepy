import numpy as np
import porepy as pp
import scipy as sp
import scipy.sparse as sps

class Up(pp.ad.Discretization):
    class _Up():
        def __init__(self, keyword):
            self.keyword = keyword
            self.upwind_matrix_key = 'upwind'
            self.dirichlet_matrix_key = 'dirichlet'
            self.segni_bordo_matrix_key = 'segni_bordo'

        def discretize(self, g, data):
            matrici = data[pp.DISCRETIZATION_MATRICES][self.keyword]
            parametri = data[pp.PARAMETERS][self.keyword]
            bc = parametri['bc']; bc_val = parametri['bc_values']
            bnd_mask = np.full(g.num_faces, False); bnd_mask[bc.bf] = True
            
            segni_facce_bordo = np.array(g.cell_faces.sum(axis=1))[:,0] * bnd_mask
            segni_flusso_darcy = np.sign(parametri['darcy_flux'])

            # sui bordi interni non voglio che la concentrazione sulla faccia venga presa da celle dentro la griglia 
            # se non lo mettessi dovrei trattare separatamente inflow e outflow... meglio cosi'!
            A = sps.diags(segni_flusso_darcy * ~bc.is_internal) * g.cell_faces 
            # +1 (flusso in accordo) * +1 (con normale uscente dalla cella) = +1 (flusso uscente)
            # -1 (flusso contrario) * -1 (a normale entrante) = +1 (flusso uscente)
            upwind = 1*(A > 0)
            matrici['upwind'] = sps.csr_matrix(upwind)

            # bordi neumann (esterni): avro' sempre darcy_flux = 0 => non mi interessa come si comporta upwind
            assert(np.allclose(segni_flusso_darcy[bc.is_neu & ~bc.is_internal], 0))

            # dirichlet inflow: ci pensa la matrice dirichlet
            # dirichlet outflow: ci pensa la matrice upwind
            # bordi interni: ci pensa segni_facce_bordo assieme alle proiezioni mortar

            # segni_bordo mi serve per fare avanti e indietro tra flusso mortar e flusso sui bordi interni
            segni_bordo = sps.diags(segni_facce_bordo)
            matrici['segni_bordo'] = segni_bordo

            bc_inflow = segni_flusso_darcy*segni_facce_bordo < 0
            dirichlet = sps.diags(1*(bc.is_dir & bc_inflow))
            matrici['dirichlet'] = dirichlet



    def __init__(self, keyword, grids):
        self.grids = grids
        self._discretization = self._Up(keyword)
        self._name = "Up"
        self.keyword = keyword

        pp.ad._ad_utils.wrap_discretization(self, self._discretization, grids=grids)

class UpCoupling(pp.ad.Discretization):
    class _UpCoupling():
        def __init__(self, keyword):
            self.keyword = keyword
            self.da_suu_matrix_key = 'da_suu'
            self.da_giu_matrix_key = 'da_giu'

        def discretize(self, g_h, g_l, data_h, data_l, data_edge):
            mg = data_edge['mortar_grid']
            matrici = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]

            parametri_h = data_h[pp.PARAMETERS][self.keyword]
            bc = parametri_h['bc']
            bnd_mask = np.full(g_h.num_faces, False); bnd_mask[bc.bf] = True

            segni_flusso_darcy = np.sign(parametri_h['darcy_flux'])
            segni_facce_bordo = np.array(g_h.cell_faces.sum(axis=1))[:,0] * bnd_mask

            laambda = mg.primary_to_mortar_int()*(segni_flusso_darcy*segni_facce_bordo)
            # TODO: riesco a mettere un assert qui sul fatto che questo lambda e' = a quello che calcolo direttamente?
            da_suu = sps.diags(1* ~(laambda < 0)) # lambda > 0 = uscente da sopra = entrante nella frattura
            da_giu = sps.diags(1*  (laambda < 0)) # lambda < 0 = entrante verso sopra = uscente dalla frattura
            matrici['da_suu'] = da_suu
            matrici['da_giu'] = da_giu

    def __init__(self, keyword, edges):
        self.edges = edges
        self._discretization = self._UpCoupling(keyword)
        self._name = "UpCoupling"
        self.keyword = keyword

        pp.ad._ad_utils.wrap_discretization(self, self._discretization, edges=edges)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from mesh import load_mesh

    nome_mesh = 'd'
    nome_sim = 'trasporto_a'

    gb = load_mesh(nome_mesh)
    darcy_flux = np.loadtxt(f"out/flussi/{nome_mesh}")

    exporter = pp.Exporter(gb, file_name=f"soluzione_{nome_sim}", folder_name=f"out/{nome_sim}/")

    rng = np.random.default_rng(10)

    for g, d in gb:
        d[pp.PRIMARY_VARIABLES] = { 'soluto': {'cells': 1}, }

        d[pp.STATE] = {}
        d[pp.STATE]['soluto'] = np.ones(g.num_cells) if g.dim == 2 else 10*np.ones(g.num_cells)

    for e, d in gb.edges():
        g = d['mortar_grid']
        d[pp.PRIMARY_VARIABLES] = {'eta': {'cells': 1}}

        d[pp.STATE] = {}
        d[pp.STATE]['eta'] = rng.random(g.num_cells)

    gs = [g for g, _ in gb]
    es = [e for e, _ in gb.edges()]
    mgs = [d['mortar_grid'] for _,d in gb.edges()]

    ds = [d for _, d in gb]
    mds = [d for _, d in gb.edges()]

    for g, d in gb:
        if not g.dim in [2]: continue

        left = pp.face_on_side(g, 'west')[0]
        right = pp.face_on_side(g, 'east')[0]
        facce_dir = np.concatenate([ left, right ])
        bc = pp.BoundaryCondition(g, faces=facce_dir, cond=np.full(facce_dir.size, 'dir'))
        bc_val = np.zeros(g.num_faces)
        bc_val[left] = 1
        bc_val[right] = 0

        # NOTE: in mass_weight non serve mettere i volumi di cella. ne tiene gia' conto MassMatrix (porepy/src/porepy/numerics/fv/mass_matrix.py:171)
        data = {'bc': bc, 'bc_values': bc_val, 'darcy_flux': np.full(g.num_faces, np.nan), 'mass_weight': np.ones(g.num_cells) }
        pp.initialize_data(g, d, 'transport', data)

    for g, d in gb:
        if not g.dim in [0, 1]: continue

        bc = pp.BoundaryCondition(g)
        bc_val = np.zeros(g.num_faces)
        data = {'bc': bc, 'bc_values': bc_val, 'darcy_flux': np.full(g.num_faces, np.nan), 'mass_weight': np.ones(g.num_cells)}
        pp.initialize_data(g, d, 'transport', data)

    for e, d in gb.edges():
        mg = d['mortar_grid']
        data = {'darcy_flux': np.full(mg.num_cells, np.nan)}
        pp.initialize_data(mg, d, 'transport', data)

    div = pp.ad.Divergence(grids=gs)
    mortar_proj = pp.ad.MortarProjections(gb=gb, grids=gs, edges=es)
    bound_ad = pp.ad.BoundaryCondition('transport', grids=gs)
    trace = pp.ad.Trace(grids=gs)
    massa = pp.ad.MassMatrixAd('transport', grids=gs)

    dof_manager = pp.DofManager(gb)

    equation_manager = pp.ad.EquationManager(gb, dof_manager)
    soluto = equation_manager.merge_variables([(g, 'soluto') for g in gs])
    eta = equation_manager.merge_variables([(e, 'eta') for e in es])

    g_dims = np.concatenate([np.full(g.num_cells, g.dim) for g in gs])
    soluto_0_ = np.select([g_dims == 0, g_dims == 1, g_dims == 2], [0, 0, 0])
    soluto_0 = pp.ad.Array(soluto_0_)

    # TODO: questo e' da fare ogni volta prima di discretizzare le matrici di upwind... quando accoppiero' darcy e trasporto da starci attenti
    flux_dof = np.concatenate([ [0], np.cumsum([g.num_faces for g in gs]) ])
    for i,d in enumerate(ds): d[pp.PARAMETERS]['transport']['darcy_flux'][:] = darcy_flux[flux_dof[i]:flux_dof[i+1]]

    up = Up('transport', gs)
    up_coupling = UpCoupling('transport', es)

    dt = 0.03

    darcy_flux_ad = pp.ad.Array(darcy_flux)
    flusso_avvettivo = darcy_flux_ad*(up.upwind*soluto + up.dirichlet*bound_ad) + up.segni_bordo*mortar_proj.mortar_to_primary_int*eta
    trasporto = (massa.mass/dt)*(soluto - soluto_0) + div*flusso_avvettivo - mortar_proj.mortar_to_secondary_int*eta

    concentrazioni_da_sopra = up_coupling.da_suu*mortar_proj.primary_to_mortar_avg*trace.trace*soluto
    concentrazioni_da_sotto = up_coupling.da_giu*mortar_proj.secondary_to_mortar_avg*soluto
    # NOTE: quando i problemi saranno accoppiati qui potro' semplicemente usare laambda al posto della prima parentesi (forse anche piu' giusto?)
    interfaccia = eta - (mortar_proj.primary_to_mortar_int*up.segni_bordo*darcy_flux_ad)*(concentrazioni_da_sopra + concentrazioni_da_sotto)

    equation_manager.equations['trasporto'] = trasporto
    equation_manager.equations['interfaccia'] = interfaccia

    # flusso_avvettivo.discretize(gb)
    # a = flusso_avvettivo.evaluate(dof_manager)

    # equation_manager.discretize(gb)
    # A, b = equation_manager.assemble()
    # soluzione = sps.linalg.spsolve(A, b)
    # dof_manager.distribute_variable(soluzione, additive=True)
    # raise SystemExit

    for i in range(100):
        equation_manager.discretize(gb)
        A, b = equation_manager.assemble()

        converged = False
        for k in range(2): # il problema e' lineare => se non converge in una iterazione c'e' qualcosa che non va
            norma_residuale = np.linalg.norm(b, np.inf)
            print(f'({i:3d}, {i*dt:.3f}, {k}): residuale: {norma_residuale:8.6f}')
            if norma_residuale < 1e-6:
                converged = True
                break

            incremento = sps.linalg.spsolve(A, b)
            if np.any(np.isnan(incremento)): break
            dof_manager.distribute_variable(incremento, additive=True)

            equation_manager.discretize(gb)
            A, b = equation_manager.assemble()

        if not converged:
            raise SystemError

        exporter.write_vtu(['soluto'], time_step=i)
        soluto_0._values = soluto.evaluate(dof_manager).val

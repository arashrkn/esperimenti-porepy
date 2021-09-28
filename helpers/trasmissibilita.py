import numpy as np
import porepy as pp
from porepy.numerics.ad.forward_mode import Ad_array
import scipy.sparse as sps
import matplotlib.pyplot as plt
from .vari import azzera_riga

def trasmissibilita_da_permeabilita_(g):
    facce, celle, segno = sps.find(g.cell_faces)
    num_mezze_facce = facce.size
    mezze_facce = np.arange(num_mezze_facce)

    if isinstance(g, pp.CartGrid):
        asdf = np.ones(num_mezze_facce)
    else:
        raggi = g.face_centers[:,facce] - g.cell_centers[:,celle]
        normali_uscenti = g.face_normals[:,facce] * segno
        
        raggi /= np.linalg.norm(raggi, axis=0)
        normali_uscenti /= np.linalg.norm(normali_uscenti, axis=0)
        
        asdf = np.einsum('ij,ij->j',raggi,normali_uscenti)

    # TODO: Delle aree ne tiene conto GeneralTpfaAd? Per far quadrare i conti sembra di si'
    # print("La trasmissivita' include le aree"); A = sps.coo_matrix((g.face_areas[facce] * asdf, (mezze_facce, celle)))
    A = sps.coo_matrix((asdf, (mezze_facce, celle)))

    avg = sps.coo_matrix((np.ones(num_mezze_facce), (facce, mezze_facce)))
    avg = sps.csr_matrix(avg / avg.sum(axis=1))

    def trasmissibilita_da_permeabilita(permeabilita):
        mezze_trasmissibilita = A * permeabilita
        trasmissibilita = (avg * mezze_trasmissibilita**-1)**-1
        return trasmissibilita
    return trasmissibilita_da_permeabilita

if __name__ == '__main__':
    from vari import azzera_riga
    from tpfa import TPFA

    X = 10
    Y = 10
    gb = pp.meshing.cart_grid([], [20, 20], physdims=[X, Y])
    g, d = [(g, d) for g, d in gb][0]

    facce_bordo = g.tags["domain_boundary_faces"].nonzero()[0]
    gamma1 = g.face_centers[0, facce_bordo] == 0
    gamma2 = g.face_centers[1, facce_bordo] == 0
    gamma3 = g.face_centers[0, facce_bordo] == X
    gamma4 = g.face_centers[1, facce_bordo] == Y

    valori_bc = np.zeros(g.num_faces)
    tipi_bc = np.full(facce_bordo.size, 'nan')

    tipi_bc[gamma1] = 'dir'; valori_bc[facce_bordo[gamma1]] = 20
    tipi_bc[gamma2] = 'neu'; valori_bc[facce_bordo[gamma2]] = 0
    tipi_bc[gamma3] = 'dir'; valori_bc[facce_bordo[gamma3]] = 10
    tipi_bc[gamma4] = 'neu'; valori_bc[facce_bordo[gamma4]] = 0

    bc = pp.BoundaryCondition(g, facce_bordo, tipi_bc)

    p_0 = Ad_array(np.zeros(g.num_cells), sps.identity(g.num_cells))
    # p = Ad_array(np.random.rand(g.num_cells), sps.identity(g.num_cells))
    # p = Ad_array(np.array([1, 0, 1, 0]), sps.identity(g.num_cells))

    porosita = np.ones(g.num_cells)
    # porosita = Ad_array(np.random.rand(g.num_cells), sps.identity(g.num_cells))
    trasmissibilita_da_permeabilita = trasmissibilita_da_permeabilita_(g)
    permeabilita = porosita ** 2
    trasmissibilita = trasmissibilita_da_permeabilita(permeabilita)
    
    tpfa = TPFA('transport', g, d)
    flusso = tpfa._apply(trasmissibilita, p_0, valori_bc, g, bc)

    div = g.cell_faces.T
    eqn = div*flusso
    p = p_0.val + sps.linalg.spsolve(eqn.jac, -eqn.val)

    print(valori_bc)
    print(p)

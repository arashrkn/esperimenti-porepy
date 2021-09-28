'''
TPFA
'''

import numpy as np
import porepy as pp
from porepy.numerics.ad.operators import Scalar, Array
import scipy.sparse as sps
from .vari import azzera_riga

class Tpfa(pp.ad.operators.ApplicableOperator):
    def __init__(self, keyword, g, data):
        self.keyword = keyword
        self.g = g
        self.data = data
        self._set_tree()

    def apply(self, trasmissibilita, potenziale, valori_bc):
        keyword = self.keyword
        g = self.g
        data = self.data
        bc = data[pp.PARAMETERS][keyword]['bc']

        flusso = self._apply(trasmissibilita, potenziale, valori_bc, g, bc)
        return flusso

    def _apply(self, trasmissibilita, potenziale, valori_bc, g, bc):
        trasmissibilita_ = trasmissibilita
        # trasmissibilita_ = trasmissibilita * g.face_areas

        div = g.cell_faces.T
        grad = -div.T

        g_p = valori_bc * bc.is_dir
        g_p_segnato = g_p * (-np.array(div.sum(axis=0))[0] * bc.is_dir)

        flusso_interno = -grad*potenziale * trasmissibilita_
        flusso_dirichlet = trasmissibilita_ * g_p_segnato
        flusso_senza_neumann = flusso_interno + flusso_dirichlet

        flusso = flusso_senza_neumann.copy()
        flusso.val[bc.is_neu] = valori_bc[bc.is_neu]
        for riga in bc.is_neu.nonzero()[0]: azzera_riga(flusso.jac.tocsr(), riga)

        return flusso

import numpy as np
import scipy.sparse as sps
import porepy as pp

# https://stackoverflow.com/questions/12129948/scipy-sparse-set-row-to-zeros
def azzera_riga(csr, riga):
    if not isinstance(csr, sps.csr_matrix): raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[riga]:csr.indptr[riga+1]] = 0

def clamp(v):
    ad = isinstance(v, pp.ad.Ad_array)
    v_val = v.val if ad else v
    neg, pos = v_val < 0, v_val > 1

    v_val[neg] = 0; v_val[pos] = 1

    if ad:
        jac = v.jac.tocsr()
        for riga in neg.nonzero()[0]: azzera_riga(jac, riga)
        for riga in pos.nonzero()[0]: azzera_riga(jac, riga)
        v.jac = jac.tocsc()
    
    return "clamp modifica l'input"

def p_(gb, dof):
    def p(ad):
        exp = pp.ad.Expression(ad, dof)
        res = exp.to_ad(gb)
        return res
    return p

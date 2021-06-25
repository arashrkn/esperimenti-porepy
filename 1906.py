'''
Simulazione 0d di un soluto e di un precipitato.
Termine di reazione implicito.
'''

import numpy as np
import porepy as pp
import scipy.sparse as sps
import matplotlib.pyplot as plt

DT = 0.1
TAU_R = 2
A = 1/TAU_R
B = 0.6
T = 8

soluzione = pp.ad.Ad_array(np.array([0.0, 0.0]), sps.identity(2))
soluzione_0 = np.array([1.0, 0.4])

I_s = sps.csc_matrix(np.array([1, 0]))
I_p = sps.csc_matrix(np.array([0, 1]))

def r(soluto, precipitato):
    # res = A*precipitato*(1 - (soluto/B)**2)
    res = A*(1 - (soluto/B)**2)
    return res

def eqn_soluto(soluzione, soluzione_0):
    soluto_0 = I_s*soluzione_0
    soluto = I_s*soluzione
    precipitato = I_p*soluzione

    r_ = r(soluto, precipitato)
    lhs = soluto/DT - r_
    rhs = soluto_0/DT
    res = lhs - rhs

    return res

def eqn_precipitato(soluzione, soluzione_0):
    precipitato_0 = I_p*soluzione_0
    soluto = I_s*soluzione
    precipitato = I_p*soluzione
    
    r_ = r(soluto, precipitato)
    lhs = precipitato/DT + r_
    rhs = precipitato_0/DT
    res = lhs - rhs

    return res

steps = int(T/DT)
ts = np.zeros(steps)
solutos = np.zeros(steps)
precipitatos = np.zeros(steps)

soluzione.val[:] = soluzione_0

ts[0] = 0
solutos[0] = (I_s*soluzione).val[0]
precipitatos[0] = (I_p*soluzione).val[0]

for i in range(1, steps):
    eqn = pp.ad.concatenate((
        eqn_soluto(soluzione, soluzione_0),
        eqn_precipitato(soluzione, soluzione_0)
    ))

    converged = False
    for k in range(10):
        residuale, jacobiano = (eqn.val, eqn.jac)
        jacobiano = jacobiano.tocsc()
        norma_residuale = np.max(np.abs(residuale))
        print(f'({i:2d},{k}): residuale: {norma_residuale:5.4f}')

        if norma_residuale < 1e-4:
            converged = True
            break

        incremento = sps.linalg.spsolve(jacobiano, -residuale)
        soluzione.val += incremento
        eqn = pp.ad.concatenate((
            eqn_soluto(soluzione, soluzione_0),
            eqn_precipitato(soluzione, soluzione_0)
        ))

    if not converged:
        print('😓')
        break

    ts[i] = i*DT
    solutos[i] = (I_s*soluzione).val[0]
    precipitatos[i] = (I_p*soluzione).val[0]

    soluzione_0[:] = soluzione.val


plt.plot(ts, solutos)
plt.plot(ts, precipitatos)
plt.axis([0, T, -0.5, 1.5])
plt.show()

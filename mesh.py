import numpy as np
import porepy as pp

def load_mesh(nome):
    if nome == 'A':
        frattura_1 = np.array([[0.2, 0.8], [0.5, 0.5]])
        gb = pp.meshing.cart_grid( [frattura_1], nx=np.array([10, 10]), physdims=np.array([1,1]) )

    if nome == 'B':
        frattura_1 = np.array([[0.3, 0.7], [0.5, 0.5]])
        frattura_2 = np.array([[0.1, 0.7], [0.8, 0.8]])
        frattura_3 = np.array([[0.5, 0.5], [0.2, 0.9]])
        gb = pp.meshing.cart_grid( [frattura_1, frattura_2, frattura_3], nx=np.array([20, 20]), physdims=np.array([1,1]) )

    if nome == 'C':
        p = np.array([
            [0.1, 0.4, 0.15, 0.9], 
            [0.1, 0.8, 0.7, 0.2]
        ])
        e = np.array([
            [0], 
            [1]
        ])
        domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}
        network_2d = pp.FractureNetwork2d(p, e, domain)
        mesh_args = {'mesh_size_frac': 0.2, 'mesh_size_bound': 0.3}
        gb = network_2d.mesh(mesh_args)

    if nome == 'D':
        p = np.array([
            [0.1, 0.4, 0.2, 0.9, 0.4, 0.8], 
            [0.1, 0.8, 0.7, 0.2, 0.3, 0.6]
        ])
        e = np.array([
            [0, 2, 4], 
            [1, 3, 5]
        ])
        domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}
        network_2d = pp.FractureNetwork2d(p, e, domain)
        mesh_args = {'mesh_size_frac': 0.05, 'mesh_size_bound': 0.05}
        gb = network_2d.mesh(mesh_args)

    if nome == 'E':
        frattura_1 = np.array([ [0.5, 0.5], [0.2, 0.8] ])
        gb = pp.meshing.cart_grid( [frattura_1], nx=np.array([20, 20]), physdims=np.array([1,1]) )

    return gb

if __name__ == '__main__':
    gb = load_mesh('D')
    pp.plot_grid(gb)

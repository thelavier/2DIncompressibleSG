import numpy as np

def get_remapped_seeds(box, Z, PeriodicX, PeriodicY):
    """
    A function that remaps the seeds so that they remain in the periodic domain

    Inputs:
        box: the fluid domain given as a list [xmin, ymin, xmax, ymax]
        Z: the seed positions
        PeriodicX: a boolian specifying periodicity in x
        PeriodicY: a boolian specifying periodicity in y
    
    Outputs:
        Z: the seeds remaped to be inside the domain

    """
    
    p = [PeriodicX, PeriodicY]
    
    bxDims = [box[2] - box[0], box[3] - box[1]]  # domain dimensions
    Binv = np.diag(p) / np.array(bxDims)     # 2x2 matrix to normalize seeds in periodic directions
    
    # Get n x 2 matrix
    # k = argmin_{l}|Z + l*diag(bxDims)|,
    # where the minimum is taken over all n x 2 matrices with integer
    # entries and n is the number of seeds
    k = np.floor(-np.dot(Z, Binv) + 0.5 * np.ones((1, 2)))
    
    Z = Z + np.dot(k, np.diag(bxDims))
    
    return Z

def zero_y_component(Z, i):
    """
    A function that zero's out the y component of the seeds for the ODE solver

    Inputs:
        Z: the seed positions
        i: the index of the solver
    
    Outputs:
        Zmod: the seed positions modified to have their y component zerod out
    """
    Zmod = Z[i].copy()
    Zmod[:, 1] = 0
    return Zmod
import numpy as np
import auxfunctions as aux

#Construct physical initial conditions as done by Charles Egan
def create_inital(perttype, g, s, f, th0, L, H, N, a, numCols, periodicx, periodicy):
    """
    Function for initialising the Geometric method of Cullen & Purser (1984) for solving the Lagrangian SG Eady slice equations with a small perturbation of the shear flow steady state defined in line 78.

    The first input variable should be one of the four strings 'unstable', 'Cullen', 'Visram' or 'stable', which specify the perturbation as that of Williams (1967), Cullen (2007), Visram et. al. (2014), or the stable perturbation of Egan et. al. (2022), respectively.
    
    If the first input varibale is 'unstable' then the domain height H is redefined to maximise the growth rate of the perturbation.
    
    If the first input varibale is 'stable' then the domain height H is redefined to ensure that the parameters are in the stable regime.

    Inputs:
        perttype: one of the four strings 'unstable', 'Cullen', 'Visram' or 'stable'
        g: double; acceleration due to gravity
        s: double; latitudinal temperature gradient
        f: double; coriolis parameter
        th0: double; reference potential temperature
        L: double; half-domain length
        H: double; domain height
        N: double; bouyancy frequency
        a: double; amplitude of perturbation
        numCols: double; desired number of columns of seeds
        periodicx: boolian; indicates if the domain is periodic in the x direction
        periodicy: boolian; indicates if the domain is periodic in the y direction

    Outputs:
        Z: n x 2 array; seed locations
        M: n x 1 array; target masses
        H: double; domain height
    """

    if perttype == 'unstable':
        H = aux.getOptimalAspectRatio(g, s, f, th0, L, N)[0]
        DPPert = lambda x1, x2: aux.DPWilliams(g, f, th0, L, H, N, a, x1, x2)

    elif perttype == 'stable':
        phase = 1 / 8
        H = aux.getStableAspectRatio(g, s, f, th0, L, N, phase)[0]
        DPPert = lambda x1, x2: aux.DPStable(g, f, th0, L, H, N, a, x1, x2)

    elif perttype == 'Cullen':
        DPPert = lambda x1, x2: aux.DPCullen(g, f, th0, L, H, a, x1, x2)

    elif perttype == 'Visram':
        DPPert = lambda x1, x2: aux.DPWilliams(g, f, th0, L, H, N, a, x1, x2 / np.pi)

    else:
        raise ValueError("The first input variable should be one of the following strings: unstable, stable, Cullen, Visram.")

    def DP(x1, x2):
        return [x1, (N ** 2 / f ** 2) * (x2 + H / 2)] + DPPert(x1, x2)

    R = [-L, 0, L, (N ** 2 / f ** 2) * H]
    delta = (R[2] - R[0]) / numCols
    Z0 = aux.getTriLattice(R, delta)

    bx = [-L, -H / 2, L, H / 2]
    X = np.column_stack([Z0[:, 0], (f ** 2 / N ** 2) * Z0[:, 1] - H / 2])
    Z = aux.get_remapped_seeds(bx, DP(X[:, 0], X[:, 1]).T, periodicx, periodicy)

    return Z, bx
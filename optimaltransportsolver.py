import numpy as np
from pysdot import OptimalTransport
from pysdot.domain_types import ConvexPolyhedraAssembly

#Constructs a domain to be passed to the laguerre functions
def make_domain(box, PeriodicX, PeriodicY):
    """
    Constructs a domain for the optimal transport problem.

    Parameters:
        box (list/tuple): Domain boundaries [xmin, ymin, zmin, xmax, ymax, zmax].
        PeriodicX, PeriodicY (bool): Periodicity flags for each axis.

    Returns:
        ConvexPolyhedraAssembly: Domain object for the optimal transport solver.
    """
    domain = ConvexPolyhedraAssembly()
    Lx, Ly = [box[i+2] - box[i] for i in range(2)]

    # Calculate the offset and size for each dimension based on periodicity
    size = [2 * Lx if PeriodicX else box[2], 
            2 * Ly if PeriodicY else box[3]]

    offset = [-Lx if PeriodicX else box[0], 
              -Ly if PeriodicY else box[1]]

    domain.add_box(offset, size)
    return domain

#Solve the Optimal transport problem and return the centroids and weights
def ot_solve(domain, Y, psi0, err_tol, PeriodicX, PeriodicY, box, solver = 'Petsc', debug = False):
    """
    Solves the optimal transport problem and returns centroids, weights, and cell masses.

    Parameters:
        domain (ConvexPolyhedraAssembly): Source domain of the optimal transport problem.
        Y (numpy.ndarray): Seed positions.
        psi0 (numpy.ndarray): Initial weight guesses.
        err_tol (float): Error tolerance for cell mass.
        PeriodicX, PeriodicY (bool): Periodicity flags.
        box (list/tuple): Domain boundaries.
        solver (str): Linear solver to use ('Petsc' or 'Scipy').
        debug (bool): Flag to enable debugging information.

    Returns:
        tuple: Centroids, optimal weights, and cell masses after optimization.
    """
    N = Y.shape[0]
    Lx, Ly = [abs(box[i+2] - box[i]) for i in range(2)]
    ot = OptimalTransport(positions=Y, weights=psi0, masses=Lx * Ly * np.ones(N) / N, domain=domain, linear_solver=solver)
    ot.set_stopping_criterion(err_tol, 'max delta masses')

    # Adding replications based on periodicity
    for x in range(-int(PeriodicX), int(PeriodicX) + 1):
        for y in range(-int(PeriodicY), int(PeriodicY) + 1):
            if x != 0 or y != 0:
                ot.pd.add_replication([Lx * x, Ly * y])

    premass = ot.get_masses() if debug else None
    ot.adjust_weights()
    psi = ot.get_weights()
    postmass = ot.pd.integrals()

    if debug:
        print('Difference in target and final mass', np.linalg.norm(premass - postmass) / np.linalg.norm(premass))

    return ot.pd.centroids(), psi, postmass
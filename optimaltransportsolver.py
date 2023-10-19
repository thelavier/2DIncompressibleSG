import numpy as np
from pysdot import OptimalTransport
from pysdot.domain_types import ConvexPolyhedraAssembly

#Constructs a domain to be passed to the laguerre functions
def make_domain(box, PeriodicX, PeriodicY, a):
    """
    Function returning the source domain for the optimal tranpsort problem.

    Inputs:
        box: list or tuple defining domain [xmin, ymin, zmin, xmax, ymax, zmax]
        img: the measure
        PeriodicX: a boolian indicating if the boundaries are periodic in x 
        PeriodicY: a boolian indicating if the boundaries are periodic in y
        a: the replication parameter

    Outputs:
        domain: domain object for passing to optimal transport solver
    """
    domain = ConvexPolyhedraAssembly()

    if PeriodicX == False and PeriodicY == False:
        domain.add_box([box[0], box[1]], [box[2], box[3]])

    elif PeriodicX == True and PeriodicY == False:
        domain.add_box([box[0] - a, box[1]], [box[2] + a, box[3]])

    elif PeriodicX == False and PeriodicY == True:
        domain.add_box([box[0], box[1] - a], [box[2], box[3] + a])

    elif PeriodicX == True and PeriodicY == True:
        domain.add_box([box[0] - a, box[1] - a], [box[2] + a, box[3] + a])

    else:
        AssertionError('Please specify periodicity.')
        
    return domain

#Solve the Optimal transport problem and return the centroids and weights
def ot_solve(domain, Y, psi0, err_tol, PeriodicX, PeriodicY, a, solver = 'Petsc', debug = False):
    """
    Function solving the optimal transport problem using the Damped Newton Method and returning the centroids and weights of the optimal diagram.

    Inputs:
        domain: The source domain of the optimal transport problem
        Y: The seed positions 
        psi0: The inital weight guess for each seed
        err_tol: The error tolerance on the mass of the cells
        PeriodicX: a boolian indicating if the boundaries are periodic in x 
        PeriodicY: a boolian indicating if the boundaries are periodic in y
        a: the replication parameter
        solver: a string that indicates which linear solver to use
        debug: a boolian indicating if the code is in debug mode

    Outputs:
        centroids: The centroids of the optimal Laguerre diagram
        psi: The optimal weights
    """
    N = Y.shape[0] #Determine the number of seeds

    if PeriodicX == False and PeriodicY == False:
        ot = OptimalTransport(positions = Y, weights = psi0, masses = domain.measure() / N * np.ones(N), domain = domain, linear_solver = solver) #Establish the Optimal Transport problem
        ot.set_stopping_criterion(err_tol, 'max delta masses') #Pick the stopping criterion to be the mass of the cells

    elif PeriodicX == True and PeriodicY == True:
        ot = OptimalTransport(positions = Y, weights = psi0, masses = 4 * np.ones(N) / N, domain = domain, linear_solver = solver) #Establish the Optimal Transport problem
        ot.set_stopping_criterion(err_tol, 'max delta masses') #Pick the stopping criterion to be the mass of the cells
        for x in [ -a, 0, a ]:
            for y in [ -a, 0, a ]:
                if x or y:
                    ot.pd.add_replication( [ x, y ] )

    elif PeriodicX == True and PeriodicY == False:
        ot = OptimalTransport(positions = Y, weights = psi0, masses = 1 * np.ones(N) / N, domain = domain, linear_solver = solver) #Establish the Optimal Transport problem
        ot.set_stopping_criterion(err_tol, 'max delta masses') #Pick the stopping criterion to be the mass of the cells
        for x in [ -a, a ]:
            ot.pd.add_replication( [ x, 0 ] )

    elif PeriodicX == False and PeriodicY == True:
        ot = OptimalTransport(positions = Y, weights = psi0, masses = 4 * np.ones(N) / N, domain = domain, linear_solver = solver) #Establish the Optimal Transport problem
        ot.set_stopping_criterion(err_tol, 'max delta masses') #Pick the stopping criterion to be the mass of the cells
        for y in [ -a, a ]:
            ot.pd.add_replication( [ 0, y ] )
    
    else:
        AssertionError('Please specify the periodicity.')

    if debug == True:
        premass = ot.get_masses()
        print('Target masses before Damped Newton', premass)
        print('Weights before Damped Newton', ot.get_weights())
        print('Mass before Damped Newton', ot.pd.integrals(), 'Total:', sum(ot.pd.integrals()))
    else:
        pass

    ot.adjust_weights() #Use Damped Newton to find the optimal weight
    psi = ot.pd.get_weights() #Extract the optimal weights from the solver

    if debug == True:
        postmass = ot.pd.integrals()
        print('Mass after Damped Newton', postmass, 'Total:', sum(postmass)) #Print the mass of each cell
        print('Difference in traget and final mass', np.linalg.norm(premass-postmass)) #Check how different the final masses are from the target masses
    else:
        pass

    return (ot.pd.centroids(), psi)


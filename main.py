import numpy as np
import optimaltransportsolver as ots
import weightguess as wg
import auxfunctions as aux

def SG_solver(Box, InitialSeeds, NumberofSeeds, PercentTolerance, FinalTime, NumberofSteps, PeriodicX, PeriodicY, a):
    """
    Function solving the Semi-Geostrophic equations using the geometric method.

    Inputs:
        box: list or tuple defining domain [xmin, ymin, zmin, xmax, ymax, zmax]
        InitialSeeds: The intial seed positions 
        NumberofSeeds: The number of seeds
        PercentTolerance: Percent tolerance, ex. 1 means 1% tolerance
        FinalTime: The end point of the simulation
        NumberofSteps: The number of steps to take to get from t=0 to t=time final
        PeriodicX: a boolian indicating if the boundaries are periodic in x 
        PeriodicY: a boolian indicating if the boundaries are periodic in y
        a: the replication parameter

        Note: The last two parameters are set up this way to integrate more easily with the animator, could be changed 

    Outputs:
        data: Outputs a saved datafile that contains the seed positions, centroid positions, and optimal weights at every timestep
    """
    #Bring parameters into the function
    box = Box
    Z0 = InitialSeeds
    N = NumberofSeeds
    per_tol = PercentTolerance
    tf = FinalTime
    Ndt = NumberofSteps

    #Construct the domain
    D = ots.make_domain(box, PeriodicX, PeriodicY, a)

    #Compute the stepsize
    dt = tf/Ndt

    #Setup extended J matrix for RHS of the ODE
    P = np.array([[0, -1], [1, 0]])
    J = np.kron(np.eye(N, dtype=int), P)

    #Setup empty data structure
    Z = np.zeros((Ndt, N, 2)) 
    C = np.zeros((Ndt, N, 2))
    w = np.zeros((Ndt, N))

    #Build the relative error tollereance 
    err_tol = ( per_tol / 100 ) * (D.measure() / N) 

    #Construct the initial state
    Z[0] = Z0
    w0 = wg.rescale_weights(box, Z[0], np.zeros(shape = (N,)), PeriodicX, PeriodicY)[0] #Rescale the weights to generate an optimized initial guess
    sol = ots.ot_solve(D, Z[0], w0, err_tol, PeriodicX, PeriodicY, a) #Solve the optimal transport problem
    C[0] = sol[0].copy() #Store the centroids
    w[0] = sol[1].copy() #Store the optimal weights

    #Use forward Euler to take an initial time step
    Zmod = aux.zero_y_component(Z, 0)
    Zint = Z[0] + dt * (J @ (np.array(Zmod - C[0]).flatten())).reshape((N, 2))
    Z[1] = aux.get_remapped_seeds(box, Zint, PeriodicX, PeriodicY)
    w0 = wg.rescale_weights(box, Z[1], np.zeros(shape = (N,)), PeriodicX, PeriodicY)[0] #Rescale the weights to generate an optimized initial guess
    sol = ots.ot_solve(D, Z[1], w0, err_tol, PeriodicX, PeriodicY, a) #Solve the optimal transport problem
    C[1] = sol[0].copy() #Store the centroids
    w[1] = sol[1].copy() #Store the optimal weights

    #Apply Adams-Bashforth 2 to solve the ODE
    for i in range(2, Ndt):

        #Use Adams-Bashforth to take a time step
        Zmod1 = aux.zero_y_component(Z, i - 1)
        Zmod2 = aux.zero_y_component(Z, i - 2)
        Zint = Z[i - 1] + (dt / 2) * (3 * J @ (np.array(Zmod1 - C[i - 1]).flatten()) - J @ (np.array(Zmod2 - C[i - 2]).flatten())).reshape((N, 2))
        Z[i] = aux.get_remapped_seeds(box, Zint, PeriodicX, PeriodicY)

        #Rescale the weights to generate an optimized initial guess
        w0 = wg.rescale_weights(box, Z[i], np.zeros(shape = (N,)), PeriodicX, PeriodicY)[0]

        #Solve the optimal transport problem
        sol = ots.ot_solve(D, Z[i], w0, err_tol, PeriodicX, PeriodicY, a)
        C[i] = sol[0].copy()

        #Save the optimal weights
        w[i] = sol[1].copy()

        print(i) #Use for tracking progress of the code when debugging.

    #Save the data
    np.savez('SG_data.npz', data1 = Z, data2 = C, data3 = w)

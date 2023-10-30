import numpy as np
import optimaltransportsolver as ots
import weightguess as wg
import auxfunctions as aux
import csv 
import os

def SG_solver(Box, InitialSeeds, PercentTolerance, FinalTime, NumberofSteps, PeriodicX, PeriodicY, mass, solver = 'Petsc', debug = False):
    """
    Function solving the Semi-Geostrophic equations using the geometric method.

    Inputs:
        box: list or tuple defining domain [xmin, ymin, xmax, ymax]
        InitialSeeds: The intial seed positions
        PercentTolerance: Percent tolerance, ex. 1 means 1% tolerance
        FinalTime: The end point of the simulation
        NumberofSteps: The number of steps to take to get from t=0 to t=time final
        PeriodicX: a boolian indicating if the boundaries are periodic in x 
        PeriodicY: a boolian indicating if the boundaries are periodic in y
        mass: the target mass if prescribed by the physical conditions
        solver: a string indicating which linear solver to use when solving the optimal transport problem
        debug: a boolian indicating if the code is in debug mode

        Note: The last two parameters are set up this way to integrate more easily with the animator, could be changed 

    Outputs:
        data: Outputs a saved datafile that contains the seed positions, centroid positions, and optimal weights at every timestep
    """
    # Bring parameters into the function
    box = Box
    Z0 = InitialSeeds
    N = int(len(Z0))
    per_tol = PercentTolerance
    tf = FinalTime
    Ndt = NumberofSteps

    # Delete the CSV file if it exists to start fresh
    if os.path.exists('./data/SG_data.csv'):
        os.remove('./data/SG_data.csv')

    # Open the CSV file for writing and create the header
    with open('./data/SG_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['time_step', 'Seeds', 'Centroids', 'Weights']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader() 

    # Construct the domain
    D = ots.make_domain(box, PeriodicX, PeriodicY)

    # Compute the stepsize
    dt = tf/Ndt

    # Setup extended J matrix for RHS of the ODE
    P = np.array([[0, -1], [1, 0]])
    J = np.kron(np.eye(N, dtype=int), P)

    # Build the relative error tollereance 
    err_tol = ( per_tol / 100 ) * (D.measure() / N) 
    
    if debug == True:
        print("Time Step", 0) # Use for tracking progress of the code when debugging.
    else:
        pass

    # Construct the initial state
    w0 = wg.rescale_weights(box, Z0, np.zeros(shape = (N,)), PeriodicX, PeriodicY)[0] #Rescale the weights to generate an optimized initial guess
    sol = ots.ot_solve(D, Z0, w0, err_tol, PeriodicX, PeriodicY, mass, box, solver, debug) #Solve the optimal transport problem

    # Create a sliding window buffer for Z, C, and w
    Z_buffer = [Z0.copy(), Z0.copy(), Z0.copy()]
    C_buffer = [sol[0].copy(), sol[0].copy(), sol[0].copy()]
    w_buffer = [sol[1].copy(), sol[1].copy(), sol[1].copy()]

    if debug == True:
        print("Time Step", 1) # Use for tracking progress of the code when debugging.
    else:
        pass

    # Use forward Euler to take an initial time step
    Zmod = aux.zero_y_component(Z_buffer, 0) #Zero out the y component for the ode solver
    Zint = Z_buffer[0] + dt * (J @ (np.array(Zmod - C_buffer[0]).flatten())).reshape((N, 2)) #Use forward Euler
    Z_buffer[1] = aux.get_remapped_seeds(box, Zint, PeriodicX, PeriodicY) #Remap the seeds to lie in the domain

    w0 = wg.rescale_weights(box, Z_buffer[1], np.zeros(shape = (N,)), PeriodicX, PeriodicY)[0] #Rescale the weights to generate an optimized initial guess
    sol = ots.ot_solve(D, Z_buffer[1], w0, err_tol, PeriodicX, PeriodicY, mass, box, solver, debug) #Solve the optimal transport problem

    C_buffer[1] = sol[0].copy() #Store the centroids
    w_buffer[1] = sol[1].copy() #Store the optimal weights

    # Save the data for time step 0 and 1
    with open('./data/SG_data.csv', 'a', newline='') as csvfile:
        fieldnames = ['time_step', 'Seeds', 'Centroids', 'Weights']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'time_step': 0,
            'Seeds': Z_buffer[0].tolist(),
            'Centroids': C_buffer[0].tolist(),
            'Weights': w_buffer[0].tolist(),
        })
        writer.writerow({
            'time_step': 1,
            'Seeds': Z_buffer[1].tolist(),
            'Centroids': C_buffer[1].tolist(),
            'Weights': w_buffer[1].tolist(),
        })

    # Apply Adams-Bashforth 2 to solve the ODE
    for i in range(2, Ndt):

        if debug == True:
            print("Time Step", i) #Use for tracking progress of the code when debugging.
        else:
            pass

        # Use Adams-Bashforth to take a time step
        Zmod1 = aux.zero_y_component(Z_buffer, (i - 1) % 3) #Zero out the y componenent for the ode solver
        Zmod2 = aux.zero_y_component(Z_buffer, (i - 2) % 3) #Zero out the y componenent for the ode solver
        Zint = Z_buffer[(i - 1) % 3] + (dt / 2) * (3 * J @ (np.array(Zmod1 - C_buffer[(i - 1) % 3]).flatten()) - J @ (np.array(Zmod2 - C_buffer[(i - 2) % 3]).flatten())).reshape((N, 2)) #Use AB2
        Z_buffer[i % 3] = aux.get_remapped_seeds(box, Zint, PeriodicX, PeriodicY) #Remap the seeds to lie in the domain

        #Rescale the weights to generate an optimized initial guess
        w0 = wg.rescale_weights(box, Z_buffer[i % 3], np.zeros(shape = (N,)), PeriodicX, PeriodicY)[0]

        #Solve the optimal transport problem
        sol = ots.ot_solve(D, Z_buffer[i % 3], w0, err_tol, PeriodicX, PeriodicY, mass, box, solver, debug)

        #Save the centroids and optimal weights
        C_buffer[i % 3] = sol[0].copy()
        w_buffer[i % 3] = sol[1].copy()

        # Save the data for Z, C, and w continuously
        with open('./data/SG_data.csv', 'a', newline='') as csvfile:
            fieldnames = ['time_step', 'Seeds', 'Centroids', 'Weights']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'time_step': i,
                'Seeds': Z_buffer[i % 3].tolist(),
                'Centroids': C_buffer[i % 3].tolist(),
                'Weights': w_buffer[i % 3].tolist(),
            })
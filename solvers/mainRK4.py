import numpy as np
from scipy import sparse
import optimaltransportsolver as ots
import weightguess as wg
import auxfunctions as aux
import os

import msgpack

def SG_solver(box, Z0, PercentTolerance, FinalTime, Ndt, PeriodicX, PeriodicY, timescale, solver = 'Petsc', debug = False):
    """
    Solves the Semi-Geostrophic equations using the geometric method.

    Parameters:
    - box (list or tuple): Domain bounds [xmin, ymin, zmin, xmax, ymax, zmax].
    - Z0 (array): Initial seed positions.
    - PercentTolerance (float): Percentage tolerance (e.g., 1 for 1% tolerance).
    - FinalTime (float): Endpoint of the simulation time.
    - Ndt (int): Number of steps from t=0 to FinalTime.
    - PeriodicX, PeriodicY (bool): Indicates if the boundaries are periodic in x, y, z respectively.
    - timescale (float): Scales the stepsize to adjust for dimesional objects
    - solver (str): Indicates the linear solver to use ('Petsc' or 'Scipy').
    - debug (bool): Enables debugging mode.

    Returns:
    - None: Saves the seed positions, centroid positions, and optimal weights at every timestep in a MessagePack file.
    """
    # Setup and initialization
    N = len(Z0)
    dt = FinalTime / Ndt * timescale
    D = ots.make_domain(box, PeriodicX, PeriodicY) # Construct the domain
    Lx, Ly = [box[i+2] - box[i] for i in range(2)]
    err_tol = (PercentTolerance / 100) * (Lx * Ly / N)
    
    # Setup extended J matrix for RHS of the ODE
    P = np.array([[0, -1], [1, 0]])
    J = sparse.kron(sparse.eye(N, dtype=int), sparse.csr_matrix(P))

    # Delete the MessagePack file if it exists to start fresh
    if os.path.exists('./data/RK4_SG_data.msgpack'):
        os.remove('./data/RK4_SG_data.msgpack')

    # Open the MessagePack file for writing and create the header
    with open('./data/RK4_SG_data.msgpack', 'wb') as msgpackfile:
        # Define the header data
        header_data = {
            'fieldnames': ['time_step', 'Seeds', 'Centroids', 'Weights', 'Mass', 'TransportCost'],
        }

        # Write the header using MessagePack
        msgpackfile.write(msgpack.packb(header_data))

    # Open the MessagePack file for writing and write the header
    with open('./data/RK4_SG_data.msgpack', 'ab') as msgpackfile:

        if debug:
            print("Time Step 0") # Use for tracking progress of the code when debugging.

        # Construct the initial state
        Z = Z0.copy()
        w0 = wg.rescale_weights(box, Z, np.zeros(shape = (N,)), PeriodicX, PeriodicY)[0] #Rescale the weights to generate an optimized initial guess
        sol = ots.ot_solve(D, Z, w0, err_tol, PeriodicX, PeriodicY, box, solver, debug) #Solve the optimal transport problem

        # Save the data for time step 0
        msgpackfile.write(msgpack.packb({
            'time_step': 0,
            'Seeds': Z.tolist(),
            'Centroids': sol[0].tolist(),
            'Weights': sol[1].tolist(),
            'Mass': sol[2].tolist(),
            'TransportCost': sol[3].tolist(),
        }))

        # Apply RK4 Method to solve the ODE
        for i in range(1, Ndt):

            if debug:
                print(f"Time Step {i}") # Use for tracking progress of the code when debugging
                
            # k1: Slope at the current state
            Zmod1 = Z.copy()
            Zmod1[:, 1] = 0 # Zero out the y component directly
            k1 = J.dot(np.array(sol[0] - Zmod1).flatten()).reshape((N, 2))

            # Intermediate state for k2
            Z_k2 = Z + (dt/2) * k1
            Z_k2 = aux.get_remapped_seeds(box, Z_k2, PeriodicX, PeriodicY)
            w_k2 = wg.rescale_weights(box, Z_k2, np.zeros(shape=(N,)), PeriodicX, PeriodicY)[0]
            sol_k2 = ots.ot_solve(D, Z_k2, w_k2, err_tol, PeriodicX, PeriodicY, box, solver, debug)
            Zmod2 = Z_k2.copy()
            Zmod2[:, 1] = 0 # Zero out the y component directly
            k2 = J.dot(np.array(sol_k2[0] - Zmod2).flatten()).reshape((N, 2))
    
            # Intermediate state for k3
            Z_k3 = Z + (dt/2) * k2
            Z_k3 = aux.get_remapped_seeds(box, Z_k3, PeriodicX, PeriodicY)
            w_k3 = wg.rescale_weights(box, Z_k3, np.zeros(shape=(N,)), PeriodicX, PeriodicY)[0]
            sol_k3 = ots.ot_solve(D, Z_k3, w_k3, err_tol, PeriodicX, PeriodicY, box, solver, debug)
            Zmod3 = Z_k3.copy()
            Zmod3[:, 1] = 0 # Zero out the y component directly
            k3 = J.dot(np.array(sol_k3[0] - Zmod3).flatten()).reshape((N, 2))
    
            # Final state for k4
            Z_k4 = Z + dt * k3
            Z_k4 = aux.get_remapped_seeds(box, Z_k4, PeriodicX, PeriodicY)
            w_k4 = wg.rescale_weights(box, Z_k4, np.zeros(shape=(N,)), PeriodicX, PeriodicY)[0]
            sol_k4 = ots.ot_solve(D, Z_k4, w_k4, err_tol, PeriodicX, PeriodicY, box, solver, debug)
            Zmod4 = Z_k4.copy()
            Zmod4[:, 1] = 0 # Zero out the y component directly
            k4 = J.dot(np.array(sol_k4[0] - Zmod4).flatten()).reshape((N, 2))
    
            # Update Z using weighted average of k1, k2, k3, and k4
            Z = Z + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            Z = aux.get_remapped_seeds(box, Z, PeriodicX, PeriodicY)

            # Solve the optimal transport probem for the updated Z
            w0 = wg.rescale_weights(box, Z, np.zeros(shape = (N,)), PeriodicX, PeriodicY)[0]
            sol = ots.ot_solve(D, Z, w0, err_tol, PeriodicX, PeriodicY, box, solver, debug)

            # Save the data for Z, C, w, and M continuously
            msgpackfile.write(msgpack.packb({
                'time_step': i,
                'Seeds': Z.tolist(),
                'Centroids': sol[0].tolist(),
                'Weights': sol[1].tolist(),
                'Mass': sol[2].tolist(),
                'TransportCost': sol[3].tolist(),
            }))
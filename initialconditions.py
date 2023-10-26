import numpy as np
import auxfunctions as aux

#Construct initial an artificial initial condition
def create_artifical_initial(N, minx, miny, maxx, maxy, Type, pert):
    """
    Function that constructs an initial condition. Allows for different distributions on different axes.

    Inputs:
        N: The number of seeds
        maxx: The maximum position in the x direction
        maxy: The maximum position in the y direction
        maxz: The maximum position in the z direction
        minx: The minimum position in the x direction
        miny: The minimum position in the y direction
        minz: The minimum position in the z direction
        Type: Type of initial condition to generate
        pert: A number between 2 and 0 that indicates the strength of a random perturbation to the initial conditions, 1 being no perturbation

    Outputs:
        matrix: The initial seeds positions
    """

    #Compute the square root of the number of seeds to later check that we can generate a valid lattice
    croot = round(N ** (1 / 2))

    if Type == 'uniform':
        # Generate random values for the first and second columns
        col_0 = np.random.uniform(minx, maxx, size=N)
        col_1 = np.random.uniform(miny, maxy, size=N)

        # Create the matrix by concatenating the columns
        matrix = np.column_stack((col_0, col_1))

        return matrix

    elif Type == 'normal':
        # Generate random values for the first and second columns
        col_0 = np.random.normal(0, maxx, size=N)
        col_1 = np.random.normal(0, maxy, size=N)

        # Create the matrix by concatenating the columns
        matrix = np.column_stack((col_0, col_1))

        return matrix

    elif Type == 'linear':
            # Generate  values for the first and second columns
            col_0 = np.linspace(minx, maxx, N)
            col_1 = np.linspace(miny, maxy, N)
    
            # Create the matrix by concatenating the columns
            matrix = np.column_stack((col_0, col_1))
    
            return matrix

    elif Type == 'linear wsp':
        # Generate  values for the first and second columns
        col_0 = np.linspace(minx, maxx, N)
        col_1 = 2 * np.sin(np.linspace(miny, maxy, N))

        # Create the matrix by concatenating the columns
        matrix = np.column_stack((col_0, col_1))

        return matrix

    elif Type == 'lattice' and N == croot ** 2:
        # Create coordinate arrays for each dimension
        col_0 = np.linspace(minx, maxx, croot)
        col_1 = np.linspace(miny, maxy, croot)

        # Create a 3D lattice using meshgrid
        Col_0, Col_1 = np.meshgrid(col_0, col_1)

        # Combine the coordinate arrays into a single matrix
        matrix = np.column_stack((Col_0.flatten(), Col_1.flatten()))

        # Construct matrix of perturbations
        perturbation = np.random.uniform(pert, 1, size = (N, 2))

        return matrix * perturbation

    elif Type == 'lattice wsp' and N == croot ** 2:
        # Create coordinate arrays for each dimension
        col_0 = np.linspace(minx, maxx, croot)
        col_1 = np.linspace(miny, maxy, croot)

        # Create a 3D lattice using meshgrid
        Col_0, Col_1 = np.meshgrid(col_0, col_1)

        # Transform the Y corrdinates to make a sine perturbation
        Col_1 = np.sin(Col_0) * np.sin(Col_1)

        # Combine the coordinate arrays into a single matrix
        matrix = np.column_stack((Col_0.flatten(), Col_1.flatten()))

        # Construct matrix of perturbations
        perturbation = np.random.uniform(pert, 1, size = (N, 2))

        return matrix * perturbation

    else:
        raise ValueError('Please specify the type of initial condition you want to use and make sure the number of seeds can generate a valid lattice.')

#Construct physical initial conditions as done by Charles Egan
def create_physical_inital(perttype, g, s, f, th0, L, H, N, a, numCols):
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
    n = Z0.shape[0]
    w0 = np.zeros(n)
    numLloyd = 100
    perL = True
    perV = False

    #for iteration in range(numLloyd):
    #    _, _, Z0 = mexPDall_2d(R, Z0, w0, perL, perV)

    M = 0 #mexPDall_2d(R, Z0, w0, perL, perV) * (f ** 2 / N ** 2)
    X = np.column_stack([Z0[:, 0], (f ** 2 / N ** 2) * Z0[:, 1] - H / 2])
    Z = DP(X[:, 0], X[:, 1]).T

    return Z, M, H
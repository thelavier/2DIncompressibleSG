import numpy as np

#Construct initial condition
def create_initial(N, minx, miny, maxx, maxy, Type):
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

    Outputs:
        matrix: The initial seeds positions
    """

    #Compute the cubic root of the number of seeds to later check that we can generate a valid lattice
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
        perturbation = np.random.uniform(0.9, 1, size = (N, 2))

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
        perturbation = np.random.uniform(0.9, 1, size = (N, 2))

        return matrix * perturbation

    else:
        raise ValueError('Please specify the type of initial condition you want to use and make sure the number of seeds can generate a valid lattice.')

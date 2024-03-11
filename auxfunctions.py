import numpy as np
import msgpack
from scipy.optimize import fsolve
from scipy.optimize import minimize

def load_data(data):
    """
    Loads data from a MessagePack file and returns numpy arrays of seeds, centroids, weights, and mass.

    Parameters:
        data (str): Filename of the MessagePack file to load.

    Returns:
        tuple: Four numpy arrays containing seeds, centroids, weights, and mass data.
    """
    # Initialize lists for data
    seeds, centroids, weights, mass, tc = [], [], [], [], []

    # Load data from file
    with open(data, mode='rb') as msgpackfile:
        unpacker = msgpack.Unpacker(msgpackfile, raw=False)
        for row in unpacker:
            seeds.append(np.array(row.get('Seeds', []), dtype=np.float64))
            centroids.append(np.array(row.get('Centroids', []), dtype=np.float64))
            weights.append(np.array(row.get('Weights', []), dtype=np.float64))
            mass.append(np.array(row.get('Mass', []), dtype=np.float64))
            tc.append(np.array(row.get('TransportCost', []), dtype=np.float64))

    # Convert lists to numpy arrays and exclude the first entry
    Z, C, W, M, TC = map(lambda x: np.array(x[1:]), (seeds, centroids, weights, mass, tc))

    return Z, C, W, M, TC

def get_remapped_seeds(box, Z, PeriodicX, PeriodicY):
    """
    Remaps seed positions to stay within a periodic domain.

    Parameters:
        box (list or tuple): Domain boundaries [xmin, ymin, zmin, xmax, ymax, zmax].
        Z (numpy.ndarray): Seed positions.
        PeriodicX (bool): Periodicity in the x-axis.
        PeriodicY (bool): Periodicity in the y-axis.

    Returns:
        numpy.ndarray: Remapped seed positions.
    """
    if PeriodicX:
        Z[:, 0] = (Z[:, 0] - box[0]) % (box[2] - box[0]) + box[0]
    if PeriodicY:
        Z[:, 1] = (Z[:, 1] - box[1]) % (box[3] - box[1]) + box[1]

    return Z

def coth(x):
    """
    Define hyperbolic cotangent function
    """
    return 1 / np.tanh(x)

def sigmaSqFun(kappa):
    """
    Define square of growth rate function (upto multiplicative constant)
    """
    return 2 * kappa * (1 / np.tanh(2 * kappa)) - 1 - kappa**2

def getOptimalAspectRatio(g, s, f, th0, L, N):
    """
    Function to find the fluid domain height (and aspect ratio) so that the maximal growth rate is achieved by the unstable normal mode perturbation defined by the function DPWilliams.

    Input:
        g: double; acceleration due to gravity
        s: double; latitudinal temperature gradient
        f: double; coriolis parameter
        th0: double; reference potential temperature
        L: double; half-domain length
        N: double; bouyancy frequency

    Output:
        optimalH: double; fluid domain height
        optimalAspectRatio: double; fluid domain aspect ratio (length/height)
        maxGrowthRate: double; maximal growth rate of perturbation
    """

    negFun = lambda kappa: -sigmaSqFun(kappa)
    kappaMax = minimize(negFun, 0.8, method='BFGS', options={'disp': False}).x[0]

    optimalH = 2 * kappaMax * f * L / N / np.pi
    optimalAspectRatio = 2 * L / optimalH
    maxGrowthRate = -np.sqrt(sigmaSqFun(kappaMax)) * g * s / N / th0 * (24 * 60 * 60)

    return optimalH, optimalAspectRatio, maxGrowthRate

def getStableAspectRatio(g, s, f, th0, L, N, phase):
    """
    Function to find the fluid domain height (and aspect ratio) so that the normal mode perturbation defined by the function DPStable is linearly stable in the classs of normal modes perturbations.

    Input:
        g: double; acceleration due to gravity
        s: double; latitudinal temperature gradient
        f: double; coriolis parameter
        th0: double; reference potential temperature
        L: double; half-domain length
        N: double; bouyancy frequency
        phase: double; desired phase speed in domain-lengths-per-day

    Output:
        stableH: double; fluid domain height
        stableAspectRatio: double; fluid domain aspect ratio (length/height)
    """
    negFun = lambda kappa: -sigmaSqFun(kappa)
    kappaCritical = fsolve(negFun, 1.2, xtol=1e-6, maxfev=1000)

    phaseSpeedMetersPerSecond = lambda kappa: np.abs(s) * g * L / N / th0 / np.pi * np.abs(np.sqrt(sigmaSqFun(kappa)))
    secsPerDay = 24 * 60 ** 2
    domLength = 2 * L
    phaseSpeedInDomainLengthsPerDay = lambda kappa: ((secsPerDay / domLength) * phaseSpeedMetersPerSecond(kappa))

    Fun = lambda kappa: phaseSpeedInDomainLengthsPerDay(kappa) - phase
    kappaStable = float(fsolve(Fun, 1.2 * kappaCritical, xtol=1e-6, maxfev=1000))

    stableH = 2 * f * L * kappaStable / N / np.pi
    stableAspectRatio = 2 * L / stableH

    return stableH, stableAspectRatio

def DPWilliams(g, f, th0, L, H, N, a, x1, x2):
    """
    Function returning function handle for the unstable perturbation defined in Williams (1967)

    Input:
        g: double; acceleration due to gravity
        f: double; coriolis parameter
        th0: double; reference potential temperature
        L: double; half-domain length
        H: double; domain height
        N: double; bouyancy frequency
        a: double; amplitude of perturbation
        x1: double; x coordinate of seed
        x2: double; y coordeinate of seed

    Output:
        DPW: function handle; function handle for the unstable perturbation defined in Williams (1967)
    """
    Bu = N * H / (f * L)
    kappa = Bu * np.pi / 2
    A1 = kappa * coth(kappa) - 1
    sigma = np.sqrt(np.abs(kappa - np.tanh(kappa)) * np.abs(coth(kappa) - kappa))
    A2 = sigma

    thetaPert = ( a * N * th0 / g * (A1 * np.sinh(np.pi * Bu * x2 / H) * np.cos(np.pi * x1 / L) - A2 * np.cosh(np.pi * Bu * x2 / H) * np.sin(np.pi * x1 / L)))
    vPert = (-a * (A1 * np.cosh(np.pi * Bu * x2 / H) * np.sin(np.pi * x1 / L) + A2 * np.sinh(np.pi * Bu * x2 / H) * np.cos(np.pi * x1 / L)))

    DPW = np.column_stack([vPert / f, g / f ** 2 / th0 * thetaPert]).T
    return DPW

def DPStable(g, f, th0, L, H, N, a, x1, x2):
    """
    Function returning function handle for the stable perturbation defined in Egan et. al. (2022)

    Input:
        g: double; acceleration due to gravity
        f: double; coriolis parameter
        th0: double; reference potential temperature
        L: double; half-domain length
        H: double; domain height
        N: double; bouyancy frequency
        a: double; amplitude of perturbation
        x1: double; x coordinate of seed
        x2: double; y coordeinate of seed

    Output:
        DPS: function handle; function handle for the stable perturbation defined in Egan et. al. (2022)
    """
    Bu = N * H / (f * L)
    kappa = Bu * np.pi / 2
    A1 = kappa * coth(kappa) - 1
    sigma = np.sqrt(np.abs(kappa - np.tanh(kappa)) * np.abs(coth(kappa) - kappa))
    A2 = sigma

    thetaPert = ( a * N * th0 / g * np.cos(np.pi * Bu * x1 / L) * (A1 * np.sinh(np.pi * Bu * x2 / H) + A2 * np.cosh(np.pi * Bu * x2 / H)))
    vPert = (-a * np.sin(np.pi * Bu * x1 / L) * (A1 * np.cosh(np.pi * Bu * x2 / H) + A2 * np.sinh(np.pi * Bu * x2 / H)))

    DPS = np.column_stack([vPert / f, g / f ** 2 / th0 * thetaPert]).T
    return DPS

def DPCullen(g, f, th0, L, H, a, x1, x2):
    """
    Function returning function handle for the unstable perturbation defined in Cullen (2007)

    Input:
    g: double; acceleration due to gravity
    f: double; coriolis parameter
    th0: double; reference potential temperature
    L: double; half-domain length
    H: double; domain height
    a: double; amplitude of perturbation
    x1: double; x coordinate of seed
    x2: double; y coordeinate of seed

    Output:
        DPC: function handle; function handle for the unstable perturbation defined in Cullen (2007)
    """
    DPC = (g * a / th0 / f ** 2) * np.sin(np.pi * (x1 / L + x2 / H + 0.5)) * np.array([H / L * np.ones_like(x1), np.ones_like(x2)])
    return DPC

def getTriLattice(bx, delta):
    """
    Function returning coordinates X of vertices of a regular triangular lattice of side length delta in a rectangular domain bx

    Lattice is oriented parrallel to the horizontal and centered in the verticle. If delta divides the length of the box exactly then the lattice is periodic in the horizontal.

    Input:
        bx: 1 x 4 array [xmin ymin xmax ymax] specifying rectangular domain
        delta: double; side length of triangles in lattice

    Output:
        X: n x 2 array; coordinates of vertices of a triangular lattice
    """
    L = bx[2] - bx[0]
    H = bx[3] - bx[1]
    nx = int(np.floor(L / delta))
    h = (np.sqrt(3) / 2) * delta
    ny = int(np.floor(H / h))
    n = nx * ny
    X = np.zeros((n, 2))
    ex = H - (ny - 1) * h

    for j in range(1, int(np.ceil(ny / 2)) + 1):
        for i in range(1, nx + 1):
            X[2 * (j - 1) * nx + i - 1] = [(i - 1) * delta + bx[0], 2 * (j - 1) * h + ex / 2 + bx[1]]

    for j in range(1, int(np.floor(ny / 2)) + 1):
        for i in range(1, nx + 1):
            X[(2 * j - 1) * nx + i - 1] = [(i - 0.5) * delta + bx[0], (2 * j - 1) * h + ex / 2 + bx[1]]

    return X

def Properties(Z, C, m, TC, th0, f, g, box):
    """
    Computes various physical properties based on seed and centroid positions.

    Parameters:
        Z (numpy.ndarray): Seed positions.
        C (numpy.ndarray): Centroid positions.
        m (numpy.ndarray): Mass array.
        TC (numpy.ndarray): Transport cost array.
        th0 (float): Background temperature in Kelvin.
        f (float): Coriolis parameter.
        g (float): Gravity.
        box (list or tuple): Domain boundaries.

    Returns:
        tuple: Calculated Meridional Velocities, Zonal Velocities, Temperature, Total Energy, and Conservation Error.
    """
    # Local parameters
    N = 5e-3 # Bouyancy frequency

    # Compute Meridonal Velocities
    MVel = f * (Z[:, :, 0] - C[:, :, 0])

    # Compute Temperature
    T = (th0 * f ** 2) / g * Z[:, :, 1]

    totalEnergy = (f ** 2 / 2) * np.sum(TC, axis = 1) - \
                    f ** 2 * box[2] * (2 * box[3]) ** 3 / 12 + \
                    N ** 2 * box[2] * (2 * box[3]) ** 3 / 6 - \
                    (f ** 2 / 2) * np.sum(m * Z[:,:,1] ** 2, axis = 1)

    meanEnergy = np.mean(totalEnergy)

    ConservationError = (meanEnergy - totalEnergy) / meanEnergy

    return MVel, T, totalEnergy, ConservationError

def compute_normalization(box, ZRef):
    """
    Computes a normalization factor based on the domain size and reference positions.

    The normalization factor is used to scale certain calculations, such as error measures,
    to account for the size of the domain and the scale of the reference positions.

    Args:
        box (list/tuple): Domain boundaries [xmin, ymin, zmin, xmax, ymax, zmax].
        ZRef (array): Reference positions, typically a 2D numpy array where each row 
                      represents a position in space.

    Returns:
        float: A normalization factor based on the domain size and the maximum position
               magnitude in the reference positions.
    """
    Lx, Ly = box[2] - box[0], box[3] - box[1]
    return 1 / np.sqrt(np.abs(Lx * Ly) * np.max(np.max(np.abs(ZRef), axis=1)) ** 2)

def get_velocity(Z, C, f):
    """
    Calculate velocity components based on seed and centroid positions.

    Parameters:
        Z (numpy.ndarray): Seed positions.
        C (numpy.ndarray): Centroid positions.
        f (float): Coriolis parameter.

    Returns:
        numpy.ndarray: Meridional velocity
    
    Raises:
        ValueError: If an invalid velocity type is provided.
    """
    return f * (Z[:, :, 0] - C[:, :, 0])
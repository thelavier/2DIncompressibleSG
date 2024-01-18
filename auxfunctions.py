import numpy as np
import msgpack
from scipy.optimize import fsolve
from scipy.optimize import minimize

def load_data(data):
    # Initialize lists to store the loaded data
    seeds = []
    centroids = []
    weights = []
    mass = []

    # Load the data from the MessagePack file
    with open(data, mode='rb') as msgpackfile:

        # Load the remaining data
        unpacker = msgpack.Unpacker(msgpackfile, raw=False)
        for row in unpacker:
            seeds.append(np.array(row.get('Seeds', []), dtype=object).astype(np.float64))
            centroids.append(np.array(row.get('Centroids', []), dtype=object).astype(np.float64))
            weights.append(np.array(row.get('Weights', []), dtype=object).astype(np.float64))
            mass.append(np.array(row.get('Mass', []), dtype=object).astype(np.float64))

    # Exclude the first entry from each list
    seeds = seeds[1:]
    centroids = centroids[1:]
    weights = weights[1:]
    mass = mass[1:]

    # Access the individual arrays
    Z = np.array(seeds)
    C = np.array(centroids)
    W = np.array(weights)
    M = np.array(mass)

    return Z, C, W, M

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
    if PeriodicX and PeriodicY:
        # Wrap points in both x and y components
        Z[:, 0] = (Z[:, 0] - box[0]) % (box[2] - box[0]) + box[0]
        Z[:, 1] = (Z[:, 1] - box[1]) % (box[3] - box[1]) + box[1]
    elif PeriodicX:
        # Wrap points in the x-component
        Z[:, 0] = (Z[:, 0] - box[0]) % (box[2] - box[0]) + box[0]
    elif PeriodicY:
        # Wrap points in the y-component
        Z[:, 1] = (Z[:, 1] - box[1]) % (box[3] - box[1]) + box[1]
    
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

def Properties(Z, C, th0, f, g):

    # Compute Meridonal Velocities
    MVel = f * (Z[:, :, 0] - C[:, :, 0])

    # Compute Temperature
    T = (th0 * f ** 2) / g * Z[:, :, 1]

    return MVel, T
import numpy as np
from pysdot import PowerDiagram
from pysdot.domain_types import ConvexPolyhedraAssembly
import pyvista as pv
import imageio.v2 as iio
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Animate the solution to the ODE
def point_animator(data, ZorC, tf):
    """
    Function animating the data produced by the optimal transport solver.

    Inputs:
        data: The data stored by the solver, must be a string
        ZorC: Decide if you want to animate the seeds or the weights, must also be a string
        Dim: Decide if you want to animate the seeds in 2D or 3D, must be a string
        tf: The 'Final time' for the solver, used to ensure that the frames and the animation interval are not jarring

    Outputs:
        animation: An animation of the seeds or the centroids depending on user choice
    """
    #Set up the animation 
    plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
    global Z
    global C

    # Load the data
    loaded_data = np.load(data)

    # Access the individual arrays
    Z = loaded_data['data1']
    C = loaded_data['data2']

    #Establish Animation parameters
    Ndt = len(Z)

    #Create the plot
    fig = plt.figure()
    fig.set_size_inches(10, 10, True)
    ax = fig.add_subplot()

    def update(i):
        global Z
        global C

        #Update the plot
        if ZorC == 'Z':
            ax.cla()
            ax.scatter(Z[i][:,0], Z[i][:,1], c = Z[i][:,0], cmap = 'jet', edgecolor = 'none', s = 8)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-5, 5])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        elif ZorC == 'C':
            ax.cla()
            ax.scatter(C[i][:,0], C[i][:,1], color = 'blue', s = 8)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-2, 2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        else:
            print('Please specify if you want to animate the centroids or the seeds!')

    if ZorC == 'Z':
        ani = animation.FuncAnimation(fig, update, frames = Ndt, interval = tf)
        FFwriter = animation.FFMpegWriter(fps = 1000)
        ani.save('./animations/SG_Seeds_2D.gif', writer = FFwriter, dpi = 100)
    elif ZorC == 'C':
        ani = animation.FuncAnimation(fig, update, frames = Ndt, interval = tf)
        FFwriter = animation.FFMpegWriter(fps = 1000)
        ani.save('./animations/SG_Centroids_2D.gif', writer = FFwriter, dpi = 100)
    else:
        print('Please specify if you want to animate the centroids or the seeds!')

def cell_animator(data, box):

    #Construct domain
    domain = ConvexPolyhedraAssembly()
    domain.add_box([box[0], box[1]], [box[2], box[3]])

    # Load the data
    loaded_data = np.load(data)

    # Access the individual arrays
    Z = loaded_data['data1']
    C = loaded_data['data2']
    w = loaded_data['data3']

    #Compute Meridonal Velocities
    f = 1e-4
    M = [[0] * len(w[0]) for _ in range(len(w))]
    for i in range(len(w)):
        for j in range(len(w[0])):
            M[i][j] = f * (Z[i][j][0] - C[i][j][0])

    # Set up the animation parameters
    n_frames = len(w)  # Number of frames

    # Create an empty list to store frames
    frames = []

    # Generate frames for the animation
    for i in range(n_frames):

        #Draw the tessellation
        pd = PowerDiagram(positions = Z[i] , weights = w[i] , domain = domain)

        # Store the meridonal wind speeds in an array
        vols = np.array(M[i])

        # Save the results in a .vtk file
        filename = "results.vtk"
        pd.display_vtk(filename)

        # Read the data
        grid=pv.read(filename)

        # create cell data that gives the cell volumes, this allows us to colour by cell volumes
        cell_vols = vols[grid.cell_data['num'].astype(int)]
        grid.cell_data['vols']=cell_vols

        # plot the data with an automatically created plotter, for a static picture use backend='static'
        plotter = pv.Plotter(window_size=[800,800], notebook = False, off_screen=True)
        plotter.add_mesh(grid)

        # Render the frame
        plotter.show()

        # Add a headlight
        light=pv.Light(light_type='headlight')
        plotter.add_light(light)

        # Get the frame as an image array
        frame = plotter.screenshot(transparent_background=True)

        # Add the frame to the list of frames
        frames.append(frame)

    # Save the frames as an animation file
    output_file = './animations/SG_Cell_Animation.gif'
    iio.mimwrite(output_file, frames, format='gif', duration = 40)
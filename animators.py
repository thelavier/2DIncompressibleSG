import numpy as np
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
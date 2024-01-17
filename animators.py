import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import auxfunctions as aux

#Animate the solution to the ODE
def point_animator(data, ZorC, box, tf):
    """
    Function animating the data produced by the optimal transport solver.

    Inputs:
        data: The data stored by the solver, must be a string
        ZorC: Decide if you want to animate the seeds or the weights, must also be a string
        box: the fluid domain represented as [xmin, ymin, xmax, ymax]
        tf: The 'Final time' for the solver, used to ensure that the frames and the animation interval are not jarring

    Outputs:
        animation: An animation of the seeds or the centroids depending on user choice
    """
    # Set up the animation 
    plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
    global Z
    global C

    # Load the data from the file
    Z, C, _, _ = aux.load_data(data)

    # Find the max and min of the seeds so that the animation domains are appropriately sized
    Zxmax = float('-inf')
    Zxmin = float('inf')
    Zymax = float('-inf')
    Zymin = float('inf')

    for frame in Z:
        # Find min and maxs in the frame
        Zxmin_in_frame = np.min(frame[:, 0])
        Zxmax_in_frame = np.max(frame[:, 0])
        Zymin_in_frame = np.min(frame[:, 1])
        Zymax_in_frame = np.max(frame[:, 1])

        # Update the min and max values 
        Zxmin = min(Zxmin, Zxmin_in_frame)
        Zxmax = max(Zxmax, Zxmax_in_frame)
        Zymin = min(Zymin, Zymin_in_frame)
        Zymax = max(Zymax, Zymax_in_frame)

    # Establish Animation parameters
    Ndt = len(Z)

    # Create the plot
    fig = plt.figure()
    fig.set_size_inches(10, 10, True)
    ax = fig.add_subplot()

    def update(i):
        global Z
        global C

        #Update the plot
        if ZorC == 'Z':
            ax.cla()
            ax.scatter(Z[i][:,0], Z[i][:,1], color = 'blue', s = 8)
            ax.set_xlim([Zxmin, Zxmax])
            ax.set_ylim([Zymin, Zymax])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        elif ZorC == 'C':
            ax.cla()
            ax.scatter(C[i][:,0], C[i][:,1], color = 'red', s = 8)
            ax.set_xlim([box[0], box[2]])
            ax.set_ylim([box[1], box[3]])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        else:
            print('Please specify if you want to animate the centroids or the seeds!')

    if ZorC == 'Z':
        ani = animation.FuncAnimation(fig, update, frames = Ndt, interval = tf)
        FFwriter = animation.FFMpegWriter(fps = 60)
        ani.save('./animations/SG_Seeds_2D.gif', writer = FFwriter, dpi = 100)
    elif ZorC == 'C':
        ani = animation.FuncAnimation(fig, update, frames = Ndt, interval = tf)
        FFwriter = animation.FFMpegWriter(fps = 60)
        ani.save('./animations/SG_Centroids_2D.gif', writer = FFwriter, dpi = 100)
    else:
        print('Please specify if you want to animate the centroids or the seeds!')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import auxfunctions as aux

#Animate the solution to the ODE
def point_animator(data, ZorC, box, tf):
    """
    Animates the solution to the ODE using data from the optimal transport solver.

    Parameters:
        data (str): File name containing the data.
        ZorC (str): 'Z' to animate seeds, 'C' to animate weights.
        box (list or tuple): Domain definition [xmin, ymin, zmin, xmax, ymax, zmax].
        tf (int): Final time for the solver to determine frame rate.

    Returns:
        Matplotlib animation: An animation of the seeds or centroids.
    """
    # Set up the animation 
    plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'

    # Load the data from the file
    Z, C, _, _, _ = aux.load_data(data)

    # Determine animation bounds
    Z_bounds = get_animation_bounds(Z)
    C_bounds = box

    # Initialize plot
    fig, ax = initialize_plot()

    # Create and save the animation
    ani = create_animation(fig, ax, Z, C, ZorC, Z_bounds, C_bounds, tf)
    save_animation(ani, ZorC)

def get_animation_bounds(frames):
    """
    Calculate the bounds for the animation based on the data frames.
    """
    all_points = np.concatenate(frames)
    min_bounds = np.min(all_points, axis=0)
    max_bounds = np.max(all_points, axis=0)
    return [min_bounds[0], min_bounds[1], max_bounds[0], max_bounds[1]]

def initialize_plot():
    """
    Initialize the plot based on the specified dimension.
    """
    fig = plt.figure()
    fig.set_size_inches(10, 10, True)
    ax = fig.add_subplot()
    return fig, ax

def create_animation(fig, ax, Z, C, ZorC, Z_bounds, C_bounds, tf):
    """
    Create the animation object.
    """
    Ndt = len(Z)
    update_func = get_update_function(ax, Z, C, ZorC, Z_bounds, C_bounds)
    return animation.FuncAnimation(fig, update_func, frames=Ndt, interval=tf)

def get_update_function(ax, Z, C, ZorC, Z_bounds, C_bounds):
    """
    Returns the appropriate update function for the animation.
    """
    def update(i):
        ax.cla()
        if ZorC == 'Z':
            plot_data(ax, Z[i], Z_bounds, ZorC)
        elif ZorC == 'C':
            plot_data(ax, C[i], C_bounds, ZorC)
        else:
            raise ValueError('Invalid ZorC value. Choose "Z" for seeds or "C" for centroids.')

    return update

def plot_data(ax, data, bounds, ZorC):
    """
    Plot the data on the given axis based on the dimension.
    """
    color = 'blue' if ZorC == 'Z' else 'red'  # Set color based on ZorC
    ax.scatter(data[:,0], data[:,1], color = color, s=8)
    ax.set_xlim([bounds[0], bounds[2]])
    ax.set_ylim([bounds[1], bounds[3]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

def save_animation(ani, ZorC):
    """
    Save the animation to a file.
    """
    filename = f'./animations/SG_{"Seeds" if ZorC == "Z" else "Centroids"}.gif'
    FFwriter = animation.FFMpegWriter(fps=30)
    ani.save(filename, writer=FFwriter, dpi=100)

# Example usage
# point_animator('data_file', 'Z', [0, 0, 0, 10, 10, 10], 1000)
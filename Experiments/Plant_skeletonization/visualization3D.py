import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import tempfile
import os
from PIL import Image


def visualize_data_3d(data, colors=None, node_size=10, show_plot=True, new_figure=True, title='', k1=1, k2=1,
                      counter_plot=0,
                      figsize=(10, 8), elevation_angle=None, azimuthal_angle=None):
    """Visualizes the data in 3D space with given colors, node sizes, and title"""
    
    if counter_plot == 0 and new_figure:
        fig = plt.figure(figsize=figsize)
    else:
        fig = None
    
    if k1 != 1 or k2 != 1:
        ax = plt.subplot(int(k1), int(k2), counter_plot + 1, projection='3d')
    else:
        ax = plt.axes(projection='3d')
    
    # Set the viewpoint
    ax.view_init(elev=elevation_angle, azim=azimuthal_angle)
    
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, s=node_size, cmap='viridis', marker='o')
    
    # If colors are provided, add a colorbar to explain the colors
    if colors is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Colors')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if show_plot:
        plt.show()
    
    return fig, ax


def plot_graph_3d(T, coordinates, node_colors=None, node_size=10, show_plot=True,
                  title='', linewidth_multiplier=1, edge_color='red', k1=1, k2=1, counter_plot=0,
                  figsize=(10, 8), elevation_angle=None, azimuthal_angle=None):
    """
    Plots a graph in 3D given a sparse adjacency matrix T and node coordinates.

    Parameters:
    - T: Sparse matrix representing the adjacency of a weighted tree.
    - coordinates: Coordinates of the nodes.
    - node_colors: Colors for the nodes.
    - node_size: Size of the nodes.
    - show_plot: Whether to show the plot.
    - title: Title for the visualization.
    - linewidth_multiplier: Multiplier for edge width based on weight.
    - edge_color: Color of the edges.

    Returns:
    - fig, ax: The figure and axis objects.
    """
    
    # Ensure T is in COO format
    if not isinstance(T, coo_matrix):
        T = coo_matrix(T)
    
    fig, ax = visualize_data_3d(coordinates, colors=node_colors, node_size=node_size,
                                show_plot=False, title=title, k1=k1, k2=k2, counter_plot=counter_plot,
                                figsize=figsize, elevation_angle=elevation_angle, azimuthal_angle=azimuthal_angle)
    
    # Iterate over the non-zero entries in T to plot edges
    for i, j, weight in zip(T.row, T.col, T.data):
        start_point = coordinates[i]
        end_point = coordinates[j]
        ax.plot([start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]],
                color=edge_color, linewidth=weight * linewidth_multiplier, alpha=0.25)
    
    if show_plot:
        plt.show()
    
    return fig, ax


def generate_rotation_gif(T, coordinates, node_colors=None, node_size=10, pedo=3,
                          title='', edge_color='red', figsize=(10, 8),
                          num_frames=60, output_filename='rotation.gif'):
    # Create azimuthal angles for the animation frames
    azimuthal_angles = np.linspace(0, 360, num_frames)
    
    # Create a figure and axis outside the loop
    fig, ax = plot_graph_3d(T, coordinates, node_colors=node_colors, node_size=node_size,
                            show_plot=False, title=title, edge_color=edge_color,
                            figsize=figsize)
    
    # Create a temporary directory with an absolute path to store frames
    with tempfile.TemporaryDirectory(prefix='/tmp/') as tmp_dir:
        tmp_filenames = []
        
        for azimuthal_angle in azimuthal_angles:
            # Update the viewpoint without regenerating the whole figure
            ax.view_init(azim=azimuthal_angle)
            
            # Save the current frame as a temporary PNG file
            tmp_filename = os.path.join(tmp_dir, f'frame_{len(tmp_filenames):04d}.png')
            tmp_filenames.append(tmp_filename)
            fig.savefig(tmp_filename)
            plt.close(fig)
        
        # Create the GIF from the saved frames using Pillow
        images = [Image.open(tmp_filename) for tmp_filename in tmp_filenames]
        images[0].save(output_filename, save_all=True, append_images=images[1:], duration=500, loop=0)

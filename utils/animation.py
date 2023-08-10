"""
This helper script handles the visualization of simulation results using a
contour plot and animations. It employs the Ax library to manage the results of
a Bayesian optimization process and matplotlib to create the plots and
animations.

The main functionality of this script includes:
- Loading the AxClient and best parameters from a saved JSON file.
- Preparing the data for plotting, including creating contour plots for
  specific metrics.
- Generating and managing animated
plots that show the progress of evaluations side by side with precomputed
images.

The final visualization consists of three parts:
- A contour plot of a specified metric against two parameters (eta and xi),
  with consecutive evaluations highlighted.
- A standard deviation plot for the specified metric.
- An animated sequence of images corresponding to the different stages of
  the simulation.

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
import matplotlib.image as mtpimg
from pathlib import Path
import os
# Ax imports
from ax.service.ax_client import AxClient
from ax.plot.contour import _get_contour_predictions

# Plot settings
matplotlib.use("TkAgg")
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['text.usetex'] = True

#                                               LOADING THE AX CLIENT FROM JSON
# =============================================================================

# Loading the AxClient
client_filename = "hex_thick_50_10_so_0_05_rd.json"
client_directory = r"C:\Users\igork\4YP\bayes-opt-for-abaqus\database"
client_filepath = os.path.abspath(
    os.path.join(client_directory, client_filename))

# Load the Ax Client from a JSON file
ax_client = AxClient.load_from_json_file(filepath=client_filepath)
best_parameters, values = ax_client.get_best_parameters()

model = ax_client.generation_strategy.model

#                                               SPECIFYING THE PLOTTED METRICS
# =============================================================================
param_x = 'eta'
param_y = 'xi'
metric_name = 'stiffness_ratio'
density = 50

#                                                   `              CONTOUR PLOT
# =============================================================================

# Creating a contour plot
data, f_plt, sd_plt, grid_x, grid_y, scales = _get_contour_predictions(
    model=model,
    x_param_name=param_x,
    y_param_name=param_y,
    metric=metric_name,
    generator_runs_dict=None,
    density=density)

X, Y = np.meshgrid(grid_x, grid_y)
Z_f = np.asarray(f_plt).reshape(density, density)
Z_sd = np.asarray(sd_plt).reshape(density, density)

labels = []
evaluations = []

for key, value in data[1].items():
    labels.append(key)
    evaluations.append(list(value[1].values()))

evaluations = np.asarray(evaluations)

#                                                                      PLOTTING
# =============================================================================
# We have 2 animated plots, side by side
fig, [ax1, ax2] = plt.subplots(nrows=1,
                               ncols=2,
                               figsize=(13.33, 7.5),
                               dpi=96,
                               gridspec_kw={'width_ratios': [1, 1.5]})

# On the right (ax2), we want to animate a precomputed list of images
# List of images to go through
image_folder = r"C:\Users\igork\4YP\photos_for_animation"
image_paths = Path(image_folder).glob('*.tif')

frames = []

for i in image_paths:
    img = mtpimg.imread(i)
    frames.append(img)

# On the left (ax1), we need plot consecutive evaluations on the contour plot
x = evaluations[:, 0]
y = evaluations[:, 1]

# Initialize two objects (one in each axes)
graph, = ax1.plot([], [], 'o', markersize=10, mfc='white', mec='black')
last_pt, = ax1.plot([], [], 'o', markersize=11, mfc='red', mec='black')
best_point, = ax1.plot([], [], '*', markersize=15, mfc='red', mec='black')

# Force equal axes
ax1.set_aspect('equal', 'box')
ax1.set(xlabel=r'$\eta$', ylabel=r'$\xi$')
ax1.set_title(r'$\frac{E_{T}}{E_{U}}$ for $\bar{\rho}=0.05$', y=1.05)

cont1 = ax1.contourf(X, Y, Z_f, 20, cmap='viridis')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(cont1, ax=ax1, cax=cax)

unit_cell = ax2.plot([], [])
ax2.set_axis_off()

no_run_frames = 10
no_slide_frames = 4
no_frames = no_run_frames + no_slide_frames


# Iterator for the animations
def animate_points(i):
    if i <= no_frames - no_slide_frames:
        graph.set_data(x[:i], y[:i])
        last_pt.set_data(x[i], y[i])
        frame = ax2.imshow(frames[i], animated=True)
    else:
        # set all points to white apart from the last
        last_pt.set_data([], [])
        graph.set_data(x[:no_run_frames], y[:no_run_frames])
        best_point.set_data(best_parameters['eta'], best_parameters['xi'])

    return graph


# for saving separate frames

# frame_dir = r"C:\Users\igork\4YP\to_send"
# for i in range(no_run_frames, no_run_frames+4):
#     animate_points(i)
#     frame_name = os.path.join(frame_dir, f'frame_{i}')
#     fig.savefig(f'{frame_name}.eps', transparent=False, bbox_inches='tight')

# Standard deviation plot
fig, ax3 = plt.subplots()

graph3, = ax3.plot([], [], 'o', markersize=10, mfc='white', mec='black')
best_point3, = ax3.plot([], [], '*', markersize=15, mfc='red', mec='black')

graph3.set_data(x[:no_run_frames], y[:no_run_frames])
best_point3.set_data(best_parameters['eta'], best_parameters['xi'])

# force equal axes
ax3.set_aspect('equal', 'box')
ax3.set(xlabel=r'$\eta$', ylabel=r'$\xi$')
ax3.set_title(
    r'Standard deviation in $\frac{E_{T}}{E_{U}}$ for $\bar{\rho}=0.05$',
    y=1.05)

cont3 = ax3.contourf(X, Y, Z_sd, 20, cmap='plasma')
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(cont3, ax=ax3, cax=cax)

plt.show()
ani = animation.FuncAnimation(fig,
                              animate_points,
                              frames=no_frames,
                              interval=100)

# Save as MP3
f = r"C:/Users/igork/4YP/to_send/contour_animation.mp4"
writervideo = animation.FFMpegWriter(fps=0.5)
ani.save(f, writer=writervideo)

plt.show()


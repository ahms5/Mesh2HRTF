# Note that this notebook do the same as the demo.py, so its up to you what
# you prefer.

# %%
# import dependencies
import mesh2scattering as m2s
import pyfar as pf
import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh

# %%
# First we need to set the project path. Here the project will be saved. If
# you want to create you own project, please change the ``project_path``.
project_path = os.path.join(
    m2s.utils.repository_root(), 'examples', 'project')

# we need to set the paths for the meshes. First for the sample and then
# for the reference plate. Please notice that the sample should lay on
# the x-y-plane where z is the hight.
path = os.path.join(
    m2s.utils.program_root(), '..',
    'tests', 'resources', 'mesh', 'sine_5k')
sample_path = os.path.join(path, 'sample.stl')
reference_path = os.path.join(path, 'reference.stl')

# Define the frequency array. For simplicity we just use 3 frequencies.
# If you want to create 3rd- or 1st octave band frequencies have a look on
# ``pyfar.dsp.filter.fractional_octave_frequencies``.
frequencies = np.array([1250, 2500, 5000])

# %%
# Now we need to define the source and receiver positions. Therefore we create
# a sampling grid, including the pole and removing the lower part of the grid.
receiver_delta_deg = 5
receiver_radius = 5
receiverPoints = pf.samplings.sph_equal_angle(
    receiver_delta_deg, receiver_radius)
receiverPoints.weights = pf.samplings.calculate_sph_voronoi_weights(receiverPoints)
receiverPoints = receiverPoints[receiverPoints.get_sph()[..., 1] < np.pi/2]
receiverPoints.show()
plt.show()

# %%
# Same for the source positions. The radius is set to 10 according to the
# diffusion ISO Standard 17497-2.
source_delta_deg = 30
source_radius = 10
sourcePoints = pf.samplings.sph_equal_angle(
    source_delta_deg, source_radius)
sourcePoints.weights = pf.samplings.calculate_sph_voronoi_weights(sourcePoints)

sourcePoints = sourcePoints[sourcePoints.get_sph()[..., 1] < np.pi/2]
sourcePoints.show()
plt.show()

# %%
# Now we need to set the parameters of the sample.
structural_wavelength = 0.177/2.5
sample_diameter = 0.8
model_scale = 2.5

# Furthermore we need to define the symmetry properties of the sample.
# In our case we have a sine-shaped surface, so the sample is symmetrical
# to the x-axe and y-axe, therefore we set the ``symmetry_azimuth`` to
# 90 and 180 degree. A rotational symmetry is not give, so we set it to
# ``False``.
symmetry_azimuth = [90, 180]
symmetry_rotational = False

# %%
# This symmetry settings are required for the postprocessing and and we can
# speed up our simulating by skipping incident angles and calculate them in
# the postprocessing by mirroring the existing data. Therefore we can skip
# the azimuth angles grater than 90 degree for the source positions.
sourcePoints_reduced = sourcePoints[sourcePoints.get_sph()[..., 0] <= np.pi/2]
sourcePoints_reduced.show()
plt.show()

# %%
# let's plot the scene
sample = trimesh.load_mesh(sample_path).vertices
sample_coords = pf.Coordinates(sample[:, 0], sample[:, 1], sample[:, 2])
ax = pf.plot.scatter(receiverPoints, s=1/72)
pf.plot.scatter(sourcePoints_reduced, ax=ax)
pf.plot.scatter(sample_coords, ax=ax, s=1/72)
plt.show()

# %%
# Now we can create the project. Please notice that the project was already
# created and simulated for demo.
m2s.input.write_scattering_project(
    project_path=project_path,
    frequencies=frequencies,
    sample_path=sample_path,
    reference_path=reference_path,
    receiver_coords=receiverPoints,
    source_coords=sourcePoints_reduced,
    structural_wavelength=structural_wavelength,
    model_scale=model_scale,
    sample_diameter=sample_diameter,
    symmetry_azimuth=symmetry_azimuth,
    symmetry_rotational=symmetry_rotational,
    )

# %%
# run project
# To execute the project you need to build the ``NumCalc`` project.
# Please follow the instruction for your operation system in the readme.
# Then you can set the path to the numcalc executable. Usually there is
# no need to change it.
numcalc_path = os.path.join(
    m2s.utils.program_root(), 'numcalc', 'bin', 'NumCalc')

# Now we can run the simulation, this may take some time. This example is
# already simulated so we don't need to wait.
m2s.numcalc.manage_numcalc(project_path, numcalc_path)

# %%
# Post processing
# Now we need to create the scattering pattern sofa files out of the
# simulation results. Here the symmetry is also applied. since the reference
# sample is always rotational symmetric, the data for the missing angles
# are rotated in this was sample and reference  data will have the same
# dimensions and coordinate
m2s.output.write_pattern(project_path)

# calculate the scattering coefficient for each incident angle and the random
# one from the scattering pattern
m2s.process.calculate_scattering(project_path)

# %%
# Read and plot data
# example of plotting the random scattering coefficient
project_name = os.path.split(project_path)[-1]
s_rand_path = os.path.join(
    project_path, f'{project_name}.scattering_rand.sofa')

s_rand, _, _ = pf.io.read_sofa(s_rand_path)
pf.plot.freq(s_rand, dB=False)
plt.show()

# %%
# plot random incidence
s_path = os.path.join(
    project_path, f'{project_name}.scattering.sofa')

s, source_pos, _ = pf.io.read_sofa(s_path)
pf.plot.freq(s, dB=False)
plt.show()

# %%

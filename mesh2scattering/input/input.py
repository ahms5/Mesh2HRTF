"""
This module provides functions to write input files for NumCalc.
"""
import os
import numpy as np
import pyfar as pf
from scipy.spatial import Delaunay, ConvexHull
import trimesh
import json
import datetime
import mesh2scattering as m2s
from packaging import version
from pathlib import Path
from enum import Enum
from .SampleMesh import SampleMesh
from .SoundSource import SoundSourceType, SoundSource
from .EvaluationGrid import EvaluationGrid


def write_scattering_project_numcalc(
        project_path:str,
        project_title:str,
        frequencies:np.ndarray,
        sound_sources:SoundSource,
        evaluation_grids:list[EvaluationGrid],
        sample_mesh:SampleMesh,
        bem_method='ML-FMM BEM',
        speed_of_sound=346.18,
        density_of_medium=1.1839,
        ):
    # check inputs
    if not isinstance(sound_sources, SoundSource):
        raise ValueError("sound_sources must be a SoundSource object.")
    if not isinstance(evaluation_grids, list) or any(
            not isinstance(i, EvaluationGrid) for i in evaluation_grids):
        raise ValueError(
            "evaluation_grids must be a list of EvaluationGrid objects.")
    if not isinstance(sample_mesh, SampleMesh):
        raise ValueError("sample_mesh must be a SampleMesh object.")
    if bem_method not in ['BEM', 'SL-FMM BEM', 'ML-FMM BEM']:
        raise ValueError(
            "method must be 'BEM', 'SL-FMM BEM', or 'ML-FMM BEM'.")
    if not isinstance(speed_of_sound, (int, float)):
        raise ValueError("speed_of_sound must be a float or int.")
    if not isinstance(density_of_medium, (int, float)):
        raise ValueError("density_of_medium must be a float or int.")
    if not isinstance(frequencies, np.ndarray):
        raise ValueError("frequencies must be a numpy array.")
    if not frequencies.ndim == 1:
        raise ValueError("frequencies must be a 1D array.")
    if not isinstance(project_title, str):
        raise ValueError("project_title must be a string.")
    if not isinstance(project_path, (str, Path)):
        raise ValueError("project_path must be a string or Path.")

    frequencies = np.array(frequencies, dtype=float)

    # create project directory if not existing
    if not os.path.isdir(project_path):
        os.mkdir(project_path)
    project_path_sample = os.path.join(project_path, 'sample')
    if not os.path.isdir(project_path_sample):
        os.mkdir(project_path_sample)
    if not os.path.isdir(os.path.join(project_path_sample, 'ObjectMeshes')):
        os.mkdir(os.path.join(project_path_sample, 'ObjectMeshes'))
    if not os.path.isdir(os.path.join(project_path_sample, 'NumCalc')):
        os.mkdir(os.path.join(project_path_sample, 'NumCalc'))
    if not os.path.isdir(os.path.join(project_path_sample, 'EvaluationGrids')):
        os.mkdir(os.path.join(project_path_sample, 'EvaluationGrids'))

    # write sample mesh
    mesh_path = os.path.join(project_path_sample, 'ObjectMeshes', 'Reference')
    sample_mesh.export_numcalc(mesh_path, start=0)
    sample_mesh.mesh.export(os.path.join(
        project_path_sample, 'ObjectMeshes', 'Reference.stl'))
    n_mesh_elements = len(sample_mesh.mesh_vertices)

    # write evaluation grid
    i_start = n_mesh_elements+100
    for evaluation_grid in evaluation_grids:
        evaluation_grid.export_numcalc(
            os.path.join(
                project_path_sample, 'EvaluationGrids', evaluation_grid.name),
            start=i_start)
        i_start += evaluation_grid.coordinates.csize

    # write NumCalc input files for all sources (NC.inp)
    version_m2s = version.parse(m2s.__version__)
    evaluation_grid_names = [grid.name for grid in evaluation_grids]
    source_type = sound_sources.source_type
    source_positions = sound_sources.source_coordinates
    n_mesh_elements = sample_mesh.n_mesh_elements
    n_mesh_nodes = sample_mesh.n_mesh_nodes
    n_grid_elements = sum([grid.faces.shape[0] for grid in evaluation_grids])
    n_grid_nodes = sum([grid.coordinates.csize for grid in evaluation_grids])
    _write_nc_inp(
        project_path_sample, version_m2s, project_title,
        speed_of_sound, density_of_medium,
        frequencies,
        evaluation_grid_names,
        source_type, source_positions,
        n_mesh_elements, n_mesh_nodes,
        n_grid_elements, n_grid_nodes,
        bem_method)

    # write parameters.json
    surface = sample_mesh.surface_description
    source_list = source_positions.cartesian.tolist()
    source_weights = source_positions.weights.tolist()
    evaluation_grids_coordinates = [
        i.coordinates.cartesian.tolist() for i in evaluation_grids]
    evaluation_grids_weights = [
        i.coordinates.weights.tolist() for i in evaluation_grids]
    parameters = {
        # project Info
        'project_title': project_title,
        'mesh2scattering_path': m2s.utils.program_root(),
        'mesh2scattering_version': m2s.__version__,
        'bem_method': bem_method,
        # Constants
        'speed_of_sound': speed_of_sound,
        'density_of_medium': density_of_medium,
        # Sample Information, post processing
        'structural_wavelength': surface.structural_wavelength_x,
        'structural_wavelength_x': surface.structural_wavelength_x,
        'structural_wavelength_y': surface.structural_wavelength_y,
        'model_scale': surface.model_scale,
        'sample_diameter': sample_mesh.sample_diameter,
        # symmetry information
        'symmetry_azimuth': surface.symmetry_azimuth,
        'symmetry_rotational': surface.symmetry_rotational,
        # frequencies
        'frequencies': list(frequencies),
        # Source definition
        'source_type': sound_sources.source_type.value,
        'sources': source_list,
        'sources_weights': source_weights,
        # Receiver definition
        'evaluation_grids_coordinates': evaluation_grids_coordinates,
        'evaluation_grids_weights': evaluation_grids_weights,

    }
    with open(os.path.join(project_path, 'parameters.json'), 'w') as file:
        json.dump(parameters, file, indent=2)


def _write_nc_inp(
        project_path: str, version: str, project_title: str,
        speed_of_sound: float, density_of_medium: float,
        frequencies: np.ndarray,
        evaluation_grid_names: list[str],
        source_type: SoundSourceType, source_positions: pf.Coordinates,
        n_mesh_elements: int, n_mesh_nodes: int,
        n_grid_elements: int, n_grid_nodes: int, method:str='ML-FMM BEM'):
    """Write NC.inp file that is read by NumCalc to start the simulation.

    The file format is documented at:
    https://github.com/Any2HRTF/Mesh2HRTF/wiki/Structure_of_NC.inp

    Parameters
    ----------
    project_path : str, Path
        root path of the NumCalc project.
    version : str
        current version of Mesh2scattering.
    project_title : str
        project title.
    speed_of_sound : float
        Speed of sound in m/s.
    density_of_medium : float
        density of the medium in kg/m^3.
    frequencies : np.ndarray
        frequency vector in Hz for NumCalc.
    evaluation_grid_names : list[str]
        evaluation grid names. Evaluation grids need to be written before the
        NC.inp file.
    source_type : SoundSourceType
        Type of the sound source. Options are 'Point source' or 'Plane wave'.
    source_positions : pf.Coordinates
        source positions in meter.
    n_mesh_elements : int
        number of elements in the mesh.
    n_mesh_nodes : int
        number of nodes in the mesh.
    n_grid_elements : int
        number of elements in the grid.
    n_grid_nodes : int
        number of nodes in the grid.
    method : str
        solving method for the NumCalc. Options are 'BEM', 'SL-FMM BEM', or
        'ML-FMM BEM'. By default 'ML-FMM BEM' is used.
    """
    materials = None
    if not isinstance(source_positions, pf.Coordinates):
        raise ValueError(
            "source_positions must be a pyfar.Coordinates object.")

    # check the BEM method
    if method == 'BEM':
        method_id = 0
    elif method == 'SL-FMM BEM':
        method_id = 1
    elif method == 'ML-FMM BEM':
        method_id = 4
    else:
        ValueError(
            f"Method must be BEM, SL-FMM BEM or ML-FMM BEM but is {method}")

    for i_source in range(source_positions.cshape[0]):

        # create directory
        filepath2 = os.path.join(
            project_path, "NumCalc", f"source_{i_source+1}")
        if not os.path.exists(filepath2):
            os.mkdir(filepath2)

        # write NC.inp
        file = open(os.path.join(filepath2, "NC.inp"), "w",
                    encoding="utf8", newline="\n")
        fw = file.write

        # header --------------------------------------------------------------
        fw("##-------------------------------------------\n")
        fw("## This file was created by mesh2scattering\n")
        fw("## Date: %s\n" % datetime.date.today())
        fw("##-------------------------------------------\n")
        fw(f"mesh2scattering {version}\n")
        fw("##\n")
        fw(f"{project_title}\n")
        fw("##\n")

        # control parameter I (hard coded, not documented) --------------------
        fw("## Controlparameter I\n")
        fw("0\n")
        fw("##\n")

        # control parameter II ------------------------------------------------
        fw("## Controlparameter II\n")
        fw("1 %d 0.000001 0.00e+00 1 0 0\n" % (
            len(frequencies)))
        fw("##\n")
        fw("## Load Frequency Curve \n")
        fw("0 %d\n" % (len(frequencies)+1))
        fw("0.000000 0.000000e+00 0.0\n")
        for ii in range(len(frequencies)):
            fw("%f %fe+04 0.0\n" % (
                0.000001*(ii+1),
                frequencies[ii] / 10000))
        fw("##\n")

        # main parameters I ---------------------------------------------------
        fw("## 1. Main Parameters I\n")
        fw("2 %d " % (n_mesh_elements+n_grid_elements))
        fw("%d 0 " % (n_mesh_nodes+n_grid_nodes))
        fw("0")
        fw(" 2 1 %s 0\n" % (method_id))
        fw("##\n")

        # main parameters II --------------------------------------------------
        fw("## 2. Main Parameters II\n")
        if "plane" in source_type.value:
            fw("1 ")
        else:
            fw("0 ")
        if "ear" in source_type.value:
            fw("0 ")
        else:
            fw("1 ")
        fw("0 0.0000e+00 0 0\n")
        fw("##\n")

        # main parameters III -------------------------------------------------
        fw("## 3. Main Parameters III\n")
        fw("0\n")
        fw("##\n")

        # main parameters IV --------------------------------------------------
        fw("## 4. Main Parameters IV\n")
        fw("%s %s 1.0\n" % (
            speed_of_sound, density_of_medium))
        fw("##\n")

        # nodes ---------------------------------------------------------------
        fw("NODES\n")
        fw("../../ObjectMeshes/Reference/Nodes.txt\n")
        # write file path of nodes to input file
        for grid in evaluation_grid_names:
            fw(f"../../EvaluationGrids/{grid}/Nodes.txt\n")
        fw("##\n")
        fw("ELEMENTS\n")
        fw("../../ObjectMeshes/Reference/Elements.txt\n")
        # write file path of elements to input file
        for grid in evaluation_grid_names:
            fw(f"../../EvaluationGrids/{grid}/Elements.txt\n")
        fw("##\n")

        # SYMMETRY ------------------------------------------------------------
        fw("# SYMMETRY\n")
        fw("# 0 0 0\n")
        fw("# 0.0000e+00 0.0000e+00 0.0000e+00\n")
        fw("##\n")

        # assign mesh elements to boundary conditions -------------------------
        # (including both, left, right ear)
        fw("BOUNDARY\n")
        # write velocity condition for the ears if using vibrating
        # elements as the sound source
        if "ear" in source_type.value:
            if i_source == 0 and \
                    source_type.value in ['Both ears', 'Left ear']:
                tmpEar = 'Left ear'
            else:
                tmpEar = 'Right ear'
            fw(f"# {tmpEar} velocity source\n")
            fw("ELEM %i TO %i VELO 0.1 -1 0.0 -1\n" % (
                materials[tmpEar]["index_start"],
                materials[tmpEar]["index_end"]))
        # remaining conditions defined by frequency curves
        curves = 0
        steps = 0
        if materials is not None:
            for m in materials:
                if materials[m]["path"] is None:
                    continue
                # write information
                fw(f"# Material: {m}\n")
                fw("ELEM %i TO %i %s 1.0 %i 1.0 %i\n" % (
                    materials[m]["index_start"],
                    materials[m]["index_end"],
                    materials[m]["boundary"],
                    curves + 1, curves + 2))
                # update metadata
                steps = max(steps, len(materials[m]["freqs"]))
                curves += 2

        fw("RETU\n")
        fw("##\n")

        # source information: point source and plane wave ---------------------
        if source_type.value == "Point source":
            fw("POINT SOURCES\n")
        elif source_type.value == "Plane wave":
            fw("PLANE WAVES\n")
        if source_type.value in ["Point source", "Plane wave"]:
            fw("0 %s %s %s 0.1 -1 0.0 -1\n" % (
                source_positions.x[i_source], source_positions.y[i_source],
                source_positions.z[i_source]))
        fw("##\n")

        # curves defining boundary conditions of the mesh ---------------------
        if curves > 0:
            fw("CURVES\n")
            # number of curves and maximum number of steps
            fw(f"{curves} {steps}\n")
            curves = 0
            for m in materials:
                if materials[m]["path"] is None:
                    continue
                # write curve for real values
                curves += 1
                fw(f"{curves} {len(materials[m]['freqs'])}\n")
                for f, v in zip(materials[m]['freqs'],
                                materials[m]['real']):
                    fw(f"{f} {v} 0.0\n")
                # write curve for imaginary values
                curves += 1
                fw(f"{curves} {len(materials[m]['freqs'])}\n")
                for f, v in zip(materials[m]['freqs'],
                                materials[m]['imag']):
                    fw(f"{f} {v} 0.0\n")

        else:
            fw("# CURVES\n")
        fw("##\n")

        # post process --------------------------------------------------------
        fw("POST PROCESS\n")
        fw("##\n")
        fw("END\n")
        file.close()



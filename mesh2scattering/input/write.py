import trimesh
import numpy as np
import os
import json
import mesh2scattering as m2s
import pyfar as pf
import datetime


def create_source_positions(phi_deg, theta_deg, radius):
    theta_rad = theta_deg * np.pi / 180.
    phi_rad = phi_deg * np.pi / 180.
    theta, phi = np.meshgrid(theta_rad, phi_rad)
    theta = theta.flatten()
    phi = phi.flatten()
    # create coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return pf.Coordinates(x, y, z)


def write_scattering_project(
        project_path, frequencies, sample_path, reference_path,
        receiver_coords, source_coords,
        structural_wavelength=0, model_scale=1, symmetry_azimuth=[90, 180],
        symmetry_rotational=False, sample_diameter=0.8,
        speed_of_sound='346.18',
        density_of_medium='1.1839'):

    if not os.path.isdir(project_path):
        os.mkdir(project_path)

    frequencyStepSize = 0
    title = 'scattering coefficient Sample'
    method = 'ML-FMM BEM'
    project_path_sample = os.path.join(project_path, 'sample')
    write_project(
        project_path_sample, title, frequencies, frequencyStepSize,
        sample_path,
        receiver_coords, source_coords, sourceType='Point source',
        method=method, materialSearchPaths=None,
        speedOfSound=speed_of_sound,
        densityOfMedium=density_of_medium, materials=None)

    title = 'scattering coefficient Reference'
    sourcePositions_ref = source_coords[
        np.abs(source_coords.get_sph()[..., 0]) < 1e-14]
    programPath = m2s.utils.repository_root()
    project_path_ref = os.path.join(project_path, 'reference')
    write_project(
        project_path_ref, title, frequencies, frequencyStepSize,
        reference_path, receiver_coords, sourcePositions_ref,
        sourceType='Point source',
        method=method,  materialSearchPaths=None,
        speedOfSound=speed_of_sound,
        densityOfMedium=density_of_medium, materials=None)

    with open(os.path.join(programPath, "..", "VERSION")) as read_version:
        version = read_version.readline()

    source_list = [list(i) for i in list(source_coords.get_cart())]
    receiver_list = [list(i) for i in list(receiver_coords.get_cart())]
    title = 'scattering pattern'
    frequencies = np.array(frequencies, dtype=float)
    parameters = {
        # project Info
        "project_title": 'scattering pattern',
        "mesh2scattering_path": m2s.utils.repository_root(),
        "mesh2scattering_version": version,
        "bem_version": 'ML-FMM BEM',
        # Constants
        "speed_of_sound": float(346.18),
        "density_of_medium": float(1.1839),
        # Sample Information, post processing
        "structural_wavelength": structural_wavelength,
        "model_scale": model_scale,
        "sample_diameter": sample_diameter,
        # symmetry information
        "symmetry_azimuth": symmetry_azimuth,
        "symmetry_rotational": symmetry_rotational,
        # frequencies
        "num_frequencies": len(frequencies),
        "min_frequency": frequencies[0],
        "max_frequency": frequencies[-1],
        "frequencies": list(frequencies),
        # Source definition
        "source_type": 'Point source',
        "sources_num": len(source_list),
        "sources": source_list,
        # Receiver definition
        "receivers_num": len(receiver_list),
        "receivers": receiver_list,

    }
    with open(os.path.join(project_path, "parameters.json"), 'w') as file:
        json.dump(parameters, file, indent=4)


def write_project(
        project_path, title, frequencies, frequencyStepSize, mesh_path,
        evaluationPoints, sourcePositions,
        sourceType='Point source', method='ML-FMM BEM',
        materialSearchPaths=None, speedOfSound='346.18',
        densityOfMedium='1.1839', materials=None):

    programPath = m2s.utils.repository_root()
    defaultPath = os.path.join(
        programPath, 'Mesh2Input', 'Materials', 'Data')
    if materialSearchPaths is None:
        materialSearchPaths = defaultPath
    else:
        materialSearchPaths += f";  {defaultPath}"

    with open(os.path.join(programPath, "..", "VERSION")) as read_version:
        version = read_version.readline()

    # create folders
    if not os.path.isdir(project_path):
        os.mkdir(project_path)
    if not os.path.isdir(os.path.join(project_path, 'ObjectMeshes')):
        os.mkdir(os.path.join(project_path, 'ObjectMeshes'))
    if not os.path.isdir(os.path.join(project_path, 'NumCalc')):
        os.mkdir(os.path.join(project_path, 'NumCalc'))
    if not os.path.isdir(os.path.join(project_path, 'EvaluationGrids')):
        os.mkdir(os.path.join(project_path, 'EvaluationGrids'))

    # write stl file
    mesh = trimesh.load(mesh_path)
    path = os.path.join(project_path, 'ObjectMeshes', 'Reference')
    write_mesh(mesh.vertices, mesh.faces, path, start=0)

    # write evaluation grid
    m2s.input.write_evaluation_grid(
        evaluationPoints,
        os.path.join(project_path, 'EvaluationGrids', 'grid'))

    # Write NumCalc input files for all sources (NC.inp) ----------------------
    _write_nc_inp(
        project_path, version, title, speedOfSound, densityOfMedium,
        frequencies, ['grid'], materials, method, sourceType,
        sourcePositions, len(mesh.faces), len(mesh.vertices))


def write_mesh(vertices, faces, path, start=200000, discard=None):
    if vertices.ndim != 2 or vertices.shape[0] < 3 \
            or vertices.shape[1] != 3:
        raise ValueError(
            "vertices must be a 2D array of shape (N, 3) with N > 2")

    # check output directory
    if not os.path.isdir(path):
        os.mkdir(path)

    # write nodes
    N = int(vertices.shape[0])
    start = int(start)

    nodes = f"{N}\n"
    for nn in range(N):
        nodes += (f"{int(start + nn)} "
                  f"{vertices[nn, 0]} "
                  f"{vertices[nn, 1]} "
                  f"{vertices[nn, 2]}\n")

    with open(os.path.join(path, "Nodes.txt"), "w") as f_id:
        f_id.write(nodes)

    # write elements
    N = int(faces.shape[0])
    elems = f"{N}\n"
    for nn in range(N):
        elems += (f"{int(start + nn)} "
                  f"{faces[nn, 0] + start} "
                  f"{faces[nn, 1] + start} "
                  f"{faces[nn, 2] + start} "
                  "0 0 0\n")

    with open(os.path.join(path, "Elements.txt"), "w") as f_id:
        f_id.write(elems)


def write_stl(mesh_path, project_path):
    mesh = trimesh.load(mesh_path)
    path = os.path.join(project_path, 'ObjectMeshes', 'Reference')
    write_mesh(mesh.vertices, mesh.faces, path, start=0)


def _write_nc_inp(filepath1, version, title,
                  speedOfSound, densityOfMedium, frequencies,
                  evaluationGrids, materials, method, sourceType,
                  sourcePositions, numElementsMesh, numNodesMesh):
    """Write NC.inp file that is read by NumCalc to start the simulation.

    The file format is documented at:
    https://github.com/Any2HRTF/Mesh2HRTF/wiki/Structure_of_NC.inp
    """
    if isinstance(sourcePositions, pf.Coordinates):
        sourcePositions = sourcePositions.get_cart()

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

    for source in range(sourcePositions.shape[0]):

        # create directory
        filepath2 = os.path.join(
            filepath1, "NumCalc", f"source_{source+1}")
        if not os.path.exists(filepath2):
            os.mkdir(filepath2)

        # write NC.inp
        file = open(os.path.join(filepath2, "NC.inp"), "w",
                    encoding="utf8", newline="\n")
        fw = file.write

        # header --------------------------------------------------------------
        fw("##-------------------------------------------\n")
        fw("## This file was created by mesh2input\n")
        fw("## Date: %s\n" % datetime.date.today())
        fw("##-------------------------------------------\n")
        fw("Mesh2HRTF %s\n" % version)
        fw("##\n")
        fw("%s\n" % title)
        fw("##\n")

        # control parameter I (hard coded, not documented) --------------------
        fw("## Controlparameter I\n")
        fw("0 0 0 0 7 0\n")
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
        numNodes = 0
        numElements = 0
        for evaluationGrid in evaluationGrids:
            # read number of nodes
            nodes = open(os.path.join(
                filepath1, "EvaluationGrids", evaluationGrid,
                "Nodes.txt"))
            line = nodes.readline()
            numNodes = numNodes+int(line)
            # read number of elements
            elements = open(os.path.join(
                filepath1, "EvaluationGrids", evaluationGrid,
                "Elements.txt"))
            line = elements.readline()
            numElements = numElements+int(line)
        fw("2 %d " % (numElementsMesh+numElements))
        fw("%d 0 " % (numNodesMesh+numNodes))
        fw("0")
        fw(" 2 1 %s 0\n" % (method_id))
        fw("##\n")

        # main parameters II --------------------------------------------------
        fw("## 2. Main Parameters II\n")
        if "plane" in sourceType:
            fw("1 ")
        else:
            fw("0 ")
        if "ear" in sourceType:
            fw("0 ")
        else:
            fw("1 ")
        fw("0 0.0000e+00 0 0 0\n")
        fw("##\n")

        # main parameters III -------------------------------------------------
        fw("## 3. Main Parameters III\n")
        fw("0 0 0 0\n")
        fw("##\n")

        # main parameters IV --------------------------------------------------
        fw("## 4. Main Parameters IV\n")
        fw("%s %se+00 1.0 0.0e+00 0.0 e+00 0.0e+00 0.0e+00\n" % (
            speedOfSound, densityOfMedium))
        fw("##\n")

        # nodes ---------------------------------------------------------------
        fw("NODES\n")
        fw("../../ObjectMeshes/Reference/Nodes.txt\n")
        # write file path of nodes to input file
        for grid in evaluationGrids:
            fw("../../EvaluationGrids/%s/Nodes.txt\n" % grid)
        fw("##\n")
        fw("ELEMENTS\n")
        fw("../../ObjectMeshes/Reference/Elements.txt\n")
        # write file path of elements to input file
        for grid in evaluationGrids:
            fw("../../EvaluationGrids/%s/Elements.txt\n" % grid)
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
        if "ear" in sourceType:
            if source == 0 and \
                    sourceType in ['Both ears', 'Left ear']:
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
        if sourceType == "Point source":
            fw("POINT SOURCES\n")
        elif sourceType == "Plane wave":
            fw("PLANE WAVES\n")
        if sourceType in ["Point source", "Plane wave"]:
            fw("0 %s %s %s 0.1 -1 0.0 -1\n" % (
                sourcePositions[source, 0], sourcePositions[source, 1],
                sourcePositions[source, 2]))
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


def _write_parameters_json(
        filepath1, title, programPath, version, method,
        evaluationGrids, materialSearchPaths, materials,
        speedOfSound, densityOfMedium,
        reference, computeHRIRs, sourceType, sourcePositions,
        frequencies, frequencyStepSize, numFrequencySteps):

    # calculate missing parameters
    sourceCenter = list(np.transpose(sourcePositions))
    sourceCenter = [list(x) for x in sourceCenter]
    sourceArea = [1]
    numSources = len(sourceCenter)

    # write parameters to dict
    parameters = {
        # project Info
        "projectTitle": title,
        "Mesh2HRTF_Path": programPath,
        "Mesh2HRTF_Version": version,
        "BEM_Type": method,
        "exportPictures": False,
        # Constants
        "speedOfSound": float(speedOfSound),
        "densityOfMedium": float(densityOfMedium),
        "3D_SceneUnit": 'm',
        # Grids and materials
        "evaluationGrids": evaluationGrids,
        "materialSearchPaths": materialSearchPaths,
        "materials": materials,
        # Source definition
        "sourceType": sourceType,
        "numSources": numSources,
        "sourceCenter": sourceCenter,
        "sourceArea": sourceArea,
        # post processing
        "reference": reference,
        "computeHRIRs": computeHRIRs,
        # frequencies
        "numFrequencies": numFrequencySteps,
        "frequencyStepSize": frequencyStepSize,
        "minFrequency": frequencies[0],
        "maxFrequency": frequencies[-1],
        "frequencies": frequencies
    }

    with open(os.path.join(filepath1, "parameters.json"), 'w') as file:
        json.dump(parameters, file, indent=4)


def _read_material_data(materials):

    for material in materials:
        # current material file
        file = materials[material]["path"]
        # check if the file exists
        if file is None:
            continue

        # initilize data
        boundary = None
        freqs = []
        real = []
        imag = []

        # read the csv material file
        with open(file, 'r') as m:
            lines = m.readlines()

        # parse the file
        for line in lines:
            line = line.strip('\n')
            # skip empty lines and comments
            if not len(line):
                continue
            if line[0] == '#':
                continue

            # detect boundary keyword
            if line in ['ADMI', 'IMPE', 'VELO', 'PRES']:
                boundary = line
            # read curve value
            else:
                line = line.split(',')
                if not len(line) == 3:
                    raise ValueError(
                        (f'Expected three values in {file} '
                         f'definition but found {len(line)}'))
                freqs.append(line[0].strip())
                real.append(line[1].strip())
                imag.append(line[2].strip())

        # check if boundary keyword was found
        if boundary is None:
            raise ValueError(
                (f"No boundary definition found in {file}. "
                 "Must be 'ADMI', 'IMPE', 'VELO', or 'PRES'"))
        # check if frequency vector is valud
        for i in range(len(freqs)-1):
            if float(freqs[i+1]) <= float(freqs[i]):
                raise ValueError((f'Frequencies in {file} '
                                  'do not increase monotonously'))

        # create output
        materials[material]['boundary'] = boundary
        materials[material]['freqs'] = freqs
        materials[material]['real'] = real
        materials[material]['imag'] = imag

    return materials

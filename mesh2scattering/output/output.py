"""Write output for Mesh2HRTF."""

import os
import warnings
import json
import numpy as np
import pyfar as pf
import glob
import sofar as sf
from mesh2scattering import utils
import csv
import re


def apply_symmetry_circular(
        data_in: pf.FrequencyData, coords_mic: pf.Coordinates,
        coords_inc: pf.Coordinates, coords_inc_out: pf.Coordinates):
    """Apply symmetry for circular symmetrical surfaces.

    Parameters
    ----------
    data_in : pf.FrequencyData
        data which is rotated, cshape need to be (#theta_coords_inc,
        #coords_mic)
    coords_mic : pf.Coordinates
        Coordinate object from the receiver positions of the current reference
        plate of cshape (#theta_coords_inc)
    coords_inc : pf.Coordinates
        Coordinate object from the source positions of the reference of cshape
        (#coords_inc_reference.
    coords_inc_out : pf.Coordinates
        Coordinate object from the source positions of the sample of cshape
        (#coords_inc_sample).

    Returns
    -------
    data_in_mirrored : pf.FrequencyData
        _description_
    """
    if not isinstance(coords_inc, pf.Coordinates):
        raise ValueError(
            'coords_inc needs to be of type pf.Coordinates, '
            f'bit it is {type(coords_inc)}.')
    if not isinstance(coords_mic, pf.Coordinates):
        raise ValueError(
            'coords_mic needs to be of type pf.Coordinates, '
            f'bit it is {type(coords_mic)}.')
    if not isinstance(coords_inc_out, pf.Coordinates):
        raise ValueError(
            'coords_inc_out needs to be of type pf.Coordinates, '
            f'bit it is {type(coords_inc_out)}.')
    if not isinstance(data_in, pf.FrequencyData):
        raise ValueError(
            'data_in needs to be of type pf.FrequencyData, '
            f'bit it is {type(data_in)}.')
    if data_in.cshape[-2] != coords_inc.csize:
        raise ValueError(
            'data_in.cshape[-2] needs to have the dimension of coords_inc '
            f'{data_in.cshape[-2]} != {coords_inc.csize}.')
    if data_in.cshape[-1] != coords_mic.csize:
        raise ValueError(
            'data_in.cshape[-1] needs to have the dimension of coords_mic '
            f'{data_in.cshape[-1]} != {coords_mic.csize}.')
    if np.max(np.abs(np.diff(coords_inc.azimuth))) > 1e-5:
        raise ValueError('coords_inc needs to have constant azimuth angle.')

    data_raw = data_in.freq.copy()
    shape = [coords_inc_out.cshape[0],
             coords_mic.cshape[0],
             len(data_in.frequencies)]
    data_out = np.empty(shape, dtype=data_in.freq.dtype)

    for idx in range(coords_inc_out.csize):
        mirrored_coord = coords_inc_out[idx].copy()
        corresponding_coord = mirrored_coord.copy()
        corresponding_coord.azimuth = coords_inc[0].azimuth
        delta_phi = -(mirrored_coord.azimuth - corresponding_coord.azimuth)
        receiver_coords_new = coords_mic.copy()
        receiver_coords_new.rotate('z', delta_phi, degrees=False)
        i, _ = coords_mic.find_nearest_k(
            receiver_coords_new.x.copy(), receiver_coords_new.y.copy(),
            receiver_coords_new.z.copy())
        i_source, _ = coords_inc.find_nearest_k(
            corresponding_coord.x, corresponding_coord.y,
            corresponding_coord.z)
        data_out[..., idx, :, :] = data_raw[..., i_source, i, :]

    data_in_mirrored = pf.FrequencyData(
        data_out, data_in.frequencies, data_in.comment)

    return data_in_mirrored


def apply_symmetry_mirror(
        data_in: pf.FrequencyData, coords_mic: pf.Coordinates,
        coords_inc: pf.Coordinates, symmetry_azimuth_deg: float):
    """Mirrors the data along a symmetry axe, defined by the azimuth_angle_deg.

    Note: Angles on the symmetry axes will be skipped for mirroring.

    Parameters
    ----------
    data_in : pf.FrequencyData
        data object of cshape (..., #incident_coords, #coords_mic),
        which should be rotated
    coords_mic : pf.Coordinates
        microphone positions
    coords_inc : pf.Coordinates
        Incident coordinates, which should be mirrored along
    symmetry_azimuth_deg : float
        azimuth angle in degree where the data should be mirrored.

    Returns
    -------
    data_in_mirrored : pf.FrequencyData
        mirrored data object for the coords_inc coordinates.
    coords_inc_mirrored : pf.Coordinates
        new coordinate object, angles on the symmetry axes will be skipped.
    """
    if not isinstance(coords_inc, pf.Coordinates):
        raise ValueError(
            'coords_inc needs to be of type pf.Coordinates, '
            f'bit it is {type(coords_inc)}.')
    if not isinstance(coords_mic, pf.Coordinates):
        raise ValueError(
            'coords_mic needs to be of type pf.Coordinates, '
            f'bit it is {type(coords_mic)}.')
    if not isinstance(data_in, pf.FrequencyData):
        raise ValueError(
            'data_in needs to be of type pf.FrequencyData, '
            f'bit it is {type(data_in)}.')
    if not isinstance(data_in, pf.FrequencyData):
        raise ValueError(
            'data needs to be of type pf.FrequencyData, '
            f'bit it is {type(data_in)}.')
    if symmetry_azimuth_deg <= 0 or symmetry_azimuth_deg >= 360:
        raise ValueError(
            '0 >= symmetry_azimuth_deg >= 360, but its '
            f'{symmetry_azimuth_deg}.')
    if data_in.cshape[-2] != coords_inc.csize:
        raise ValueError(
            'data_in.cshape[-2] needs to have the dimension of coords_inc '
            f'{data_in.cshape[-2]} != {coords_inc.csize}.')
    if data_in.cshape[-1] != coords_mic.csize:
        raise ValueError(
            'data_in.cshape[-1] needs to have the dimension of coords_mic '
            f'{data_in.cshape[-1]} != {coords_mic.csize}.')

    symmetry_angle_rad = symmetry_azimuth_deg*np.pi/180
    if np.abs(symmetry_azimuth_deg - 180) < 1e-8:
        idx = np.abs(coords_inc.azimuth % symmetry_angle_rad) > 1e-8
    elif np.abs(symmetry_azimuth_deg - 90) < 1e-8:
        idx = np.abs(coords_inc.azimuth - symmetry_angle_rad) > 1e-8
    no_top_point = coords_inc.colatitude > 1e-8
    idx = np.flip(np.where(idx & no_top_point)).flatten()
    dim_inc = -3
    coords_mirrored = coords_inc[idx].copy()
    coords_mirrored.azimuth = 2*symmetry_angle_rad - coords_mirrored.azimuth
    data_mirrored_shape = list(data_in.freq.shape)
    data_mirrored_shape[dim_inc] = len(idx)
    data_mirrored = np.zeros(data_mirrored_shape, dtype=data_in.freq.dtype)
    data_raw = data_in.freq.copy()

    for idx_mirror in range(coords_mirrored.csize):
        mirrored_coord = coords_mirrored[idx_mirror].copy()
        corresponding_coord = mirrored_coord.copy()
        corresponding_coord.azimuth \
            = 2*symmetry_angle_rad - corresponding_coord.azimuth
        delta_phi = -(mirrored_coord.azimuth - corresponding_coord.azimuth)
        receiver_coords_new = coords_mic.copy()
        receiver_coords_new.rotate('z', delta_phi, degrees=False)
        i, _ = coords_mic.find_nearest_k(
            receiver_coords_new.x.copy(), receiver_coords_new.y.copy(),
            receiver_coords_new.z.copy())
        i_source, _ = coords_inc.find_nearest_k(
            corresponding_coord.x, corresponding_coord.y,
            corresponding_coord.z)
        data_mirrored[..., idx_mirror, :, :] = data_raw[..., i_source, i, :]

    coords_ = np.concatenate(
        (coords_inc.cartesian, coords_mirrored.cartesian), axis=0).T
    if coords_inc.weights is not None:
        weights_ = np.concatenate(
            (coords_inc.weights, coords_mirrored.weights), axis=0)
    else:
        weights_ = None
    coords_inc_mirrored = pf.Coordinates(
        coords_[0], coords_[1], coords_[2], weights=weights_)
    new_data = np.concatenate((data_in.freq, data_mirrored), axis=dim_inc)
    data_in_mirrored = pf.FrequencyData(
        new_data, data_in.frequencies, data_in.comment)

    return data_in_mirrored, coords_inc_mirrored


def write_pattern(folder):
    """
    Process NumCalc output and write data to disk.

    Processing the data is done in the following steps

    1. Read project parameter `from parameters.json`
    2. use :py:func:`~write_output_report` to parse files in
       project_folder/NumCalc/source_*/NC*.out, write project report to
       project_folder/Output2HRTF/report_source_*.csv. Raise a warning if any
       issues were detected and write report_issues.txt to the same folder
    3. Read simulated pressures from project_folder/NumCalc/source_*/be.out.
       This and the following steps are done, even if an issue was detected in
       the previous step
    4. use :py:func:`~mesh2hrtf.reference_hrtfs` and
       :py:func:`~mesh2hrtf.compute_hrirs` to save the results to SOFA files

    Parameters
    ----------
    folder : str, optional
        The path of the Mesh2HRTF project folder, i.e., the folder containing
        the subfolders EvaluationsGrids, NumCalc, and ObjectMeshes. The
        default, ``None`` uses the current working directory.
    """

    if (not os.path.exists(os.path.join(folder, 'reference'))) \
            or (not os.path.exists(os.path.join(folder, 'sample'))):
        raise ValueError(
            "Folder need to contain reference and sample folders.")

    # read sample data
    evaluationGrids, params = read_numcalc(
        os.path.join(folder, 'sample'), False)

    # process BEM data for writing HRTFs and HRIRs to SOFA files
    for grid in evaluationGrids:
        print(f'\nWrite sample data "{grid}" ...\n')
        # get pressure as SOFA object (all following steps are run on SOFA
        # objects. This way they are available to other users as well)
        source_position = np.array(params["sources"])
        if source_position.shape[1] != 3:
            source_position = np.transpose(source_position)
        receiver_position = np.array(evaluationGrids[grid]["nodes"][:, 1:4])
        if receiver_position.shape[1] != 3:
            receiver_position = np.transpose(receiver_position)

        # apply symmetry of reference sample
        data_raw = np.moveaxis(evaluationGrids[grid]["pressure"], 1, 0)
        data = pf.FrequencyData(
            data_raw, params["frequencies"])
        receiver_coords = _cart_coordinates(receiver_position)
        source_coords = _cart_coordinates(source_position)
        for i in params['symmetry_azimuth']:
            data, source_coords = apply_symmetry_mirror(
                data, receiver_coords, source_coords, i)

        # write data
        sofa = utils._get_sofa_object(
            data.freq,
            source_coords.cartesian,
            receiver_position,
            params["mesh2scattering_version"],
            frequencies=params["frequencies"])

        sofa.GLOBAL_Title = folder.split(os.sep)[-1]

        # write scattered sound pressure to SOFA file
        sf.write_sofa(os.path.join(
            folder, 'sample.pattern.sofa'), sofa)

    evaluationGrids, params = read_numcalc(
        os.path.join(folder, 'reference'), True)

    # process BEM data for writing scattered sound pressure to SOFA files
    for grid in evaluationGrids:
        print(f'\nWrite sample data "{grid}" ...\n')
        # get pressure as SOFA object (all following steps are run on SOFA
        # objects. This way they are available to other users as well)
        # read source and receiver positions
        receiver_position_ref = np.array(
            evaluationGrids[grid]["nodes"][:, 1:4])
        if receiver_position_ref.shape[1] != 3:
            receiver_position_ref = np.transpose(receiver_position_ref)

        # apply symmetry of reference sample
        source_position_ref = source_coords[
            np.abs(source_coords.azimuth) < 1e-14]
        data = evaluationGrids[grid]["pressure"]
        if source_coords.csize != source_position_ref.cshape[0]:
            data = np.swapaxes(data, 0, 1)
            data = apply_symmetry_circular(
                pf.FrequencyData(data, params["frequencies"]),
                _cart_coordinates(receiver_position_ref),
                source_position_ref,
                source_coords).freq

        # create sofa file
        sofa = utils._get_sofa_object(
            data,
            source_coords.cartesian,
            receiver_position_ref,
            params["mesh2scattering_version"],
            frequencies=params["frequencies"])

        sofa.GLOBAL_Title = folder.split(os.sep)[-1]

        # write HRTF data to SOFA file
        sf.write_sofa(os.path.join(
            folder, 'reference.pattern.sofa'), sofa)

    print('Done\n')


def _cart_coordinates(xyz):
    return pf.Coordinates(xyz[:, 0], xyz[:, 1], xyz[:, 2])


def check_project(folder=None):
    r"""
    Generate project report from NumCalc output files.

    NumCalc (Mesh2HRTF's numerical core) writes information about the
    simulations to the files `NC*.out` located under `NumCalc/source_*`. The
    file `NC.out` exists if NumCalc was ran without the additional command line
    parameters ``-istart`` and ``-iend``. If these parameters were used, there
    is at least one `NC\*-\*.out`. If this is the case, information from
    `NC\*-\*.out` overwrites information from NC.out in the project report.

    .. note::

        The project reports are written to the files
        `Output2HRTF/report_source_*.csv`. If issues were detected, they are
        listed in `Output2HRTF/report_issues.csv`.

    The report contain the following information

    Frequency step
        The index of the frequency.
    Frequency in Hz
        The frequency in Hz.
    NC input
        Name of the input file from which the information was taken.
    Input check passed
        Contains a 1 if the check of the input data passed and a 0 otherwise.
        If the check failed for one frequency, the following frequencies might
        be affected as well.
    Converged
        Contains a 1 if the simulation converged and a 0 otherwise. If the
        simulation did not converge, the relative error might be high.
    Num. iterations
        The number of iterations that were required to converge
    relative error
        The relative error of the final simulation
    Comp. time total
        The total computation time in seconds
    Comp. time assembling
        The computation time for assembling the matrices in seconds
    Comp. time solving
        The computation time for solving the matrices in seconds
    Comp. time post-proc
        The computation time for post-processing the results in seconds


    Parameters
    ----------
    folder : str, optional
        The path of the Mesh2HRTF project folder, i.e., the folder containing
        the subfolders EvaluationsGrids, NumCalc, and ObjectMeshes. The
        default, ``None`` uses the current working directory.

    Returns
    -------
    found_issues : bool
        ``True`` if issues were found, ``False`` otherwise
    report : str
        The report or an empty string if no issues were found
    """

    if folder is None:
        folder = os.getcwd()

    # get sources and number of sources and frequencies
    sources = glob.glob(os.path.join(folder, "NumCalc", "source_*"))
    num_sources = len(sources)

    with open(os.path.join(folder, '..', "parameters.json"), "r") as file:
        params = json.load(file)

    # sort source files (not read in correct order in some cases)
    nums = [int(source.split("_")[-1]) for source in sources]
    sources = np.array(sources)
    sources = sources[np.argsort(nums)]

    # parse all NC*.out files for all sources
    all_files, fundamentals, out, out_names = _parse_nc_out_files(
        sources, num_sources, params["num_frequencies"])

    return all_files, fundamentals, out, out_names


def merge_frequency_data(data_list):
    """
    Merge multiple FrequencyData objects into one.

    Parameters
    ----------
    data_list : list of pyfar.FrequencyData
        List of FrequencyData objects to be merged.

    Returns
    -------
    data_out : pyfar.FrequencyData
        Merged FrequencyData object.
    """
    data_out = data_list[0].copy()
    frequencies = data_out.frequencies.copy()
    for idx in range(1, len(data_list)):
        data = data_list[idx]
        assert data_out.cshape == data.cshape
        frequencies = np.append(frequencies, data.frequencies)
        frequencies = np.array(list(set(frequencies)))
        frequencies = np.sort(frequencies)

        data_new = []
        for f in frequencies:
            if any(data_out.frequencies == f):
                freq_index = np.where(data_out.frequencies == f)
                data_new.append(data_out.freq[..., freq_index[0][0]])
            elif any(data.frequencies == f):
                freq_index = np.where(data.frequencies == f)
                data_new.append(data.freq[..., freq_index[0][0]])

        data_new = np.moveaxis(np.array(data_new), 0, -1)
        data_out = pf.FrequencyData(data_new, frequencies)
    return data_out


def read_numcalc(folder=None, is_ref=False):
    """
    Process NumCalc output and write data to disk.

    Processing the data is done in the following steps

    1. Read project parameter `from parameters.json`
    2. use :py:func:`~write_output_report` to parse files in
       project_folder/NumCalc/source_*/NC*.out, write project report to
       project_folder/Output2HRTF/report_source_*.csv. Raise a warning if any
       issues were detected and write report_issues.txt to the same folder
    3. Read simulated pressures from project_folder/NumCalc/source_*/be.out.
       This and the following steps are done, even if an issue was detected in
       the previous step
    4. use :py:func:`~mesh2hrtf.reference_hrtfs` and
       :py:func:`~mesh2hrtf.compute_hrirs` to save the results to SOFA files

    Parameters
    ----------
    folder : str, optional
        The path of the Mesh2HRTF project folder, i.e., the folder containing
        the subfolders EvaluationsGrids, NumCalc, and ObjectMeshes. The
        default, ``None`` uses the current working directory.
    is_ref : bool, optional
        If the data is from the reference sample. The default is False.
    """

    # check input
    if folder is None:
        folder = os.getcwd()

    # check and load parameters, required parameters are:
    # Mesh2HRTF_version, reference, computeHRIRs, speedOfSound, densityOfAir,
    # sources_num, sourceType, sources, sourceArea,
    # num_frequencies, frequencies
    params = os.path.join(folder, '..', 'parameters.json')
    if not os.path.isfile(params):
        raise ValueError((
            f"The folder {folder} is not a valid Mesh2scattering project. "
            "It must contain the file 'parameters.json'"))

    with open(params, "r") as file:
        params = json.load(file)

    # get source positions
    if params['sources_num'] > 1:
        source_coords = np.transpose(np.array(params['sources']))
    else:
        source_coords = np.array(params['sources']).reshape((1, 3))
    source_coords = pf.Coordinates(
        source_coords[..., 0], source_coords[..., 1], source_coords[..., 2])

    # output directory
    if not os.path.exists(os.path.join(folder, 'Output2HRTF')):
        os.makedirs(os.path.join(folder, 'Output2HRTF'))

    # write the project report and check for issues
    print('\n Writing the project report ...')
    found_issues, report = write_output_report(folder)

    if found_issues:
        warnings.warn(report, stacklevel=2)

    # get the evaluation grids
    evaluationGrids, _ = _read_nodes_and_elements(
        os.path.join(folder, 'EvaluationGrids'))

    # Load EvaluationGrid data
    if is_ref:
        xyz = np.array(params["sources"])
        coords = pf.Coordinates(xyz[..., 0], xyz[..., 1], xyz[..., 2])
        num_sources = np.sum(np.abs(coords.azimuth) < 1e-12)
    else:
        num_sources = params["sources_num"]

    if not len(evaluationGrids) == 0:
        pressure, _ = _read_numcalc_data(
            num_sources, params["num_frequencies"],
            folder, 'pEvalGrid')

    # save to struct
    cnt = 0
    for grid in evaluationGrids:
        evaluationGrids[grid]["pressure"] = pressure[
            cnt:cnt+evaluationGrids[grid]["num_nodes"], :, :]

        cnt = cnt + evaluationGrids[grid]["num_nodes"]

    receiver_coords = evaluationGrids[grid]["nodes"][:, 1:4]
    receiver_coords = pf.Coordinates(
        receiver_coords[..., 0], receiver_coords[..., 1],
        receiver_coords[..., 2])

    return evaluationGrids, params


def write_output_report(folder=None):
    r"""
    Generate project report from NumCalc output files.

    NumCalc (Mesh2HRTF's numerical core) writes information about the
    simulations to the files `NC*.out` located under `NumCalc/source_*`. The
    file `NC.out` exists if NumCalc was ran without the additional command line
    parameters ``-istart`` and ``-iend``. If these parameters were used, there
    is at least one `NC\*-\*.out`. If this is the case, information from
    `NC\*-\*.out` overwrites information from NC.out in the project report.

    .. note::

        The project reports are written to the files
        `Output2HRTF/report_source_*.csv`. If issues were detected, they are
        listed in `Output2HRTF/report_issues.csv`.

    The report contain the following information

    Frequency step
        The index of the frequency.
    Frequency in Hz
        The frequency in Hz.
    NC input
        Name of the input file from which the information was taken.
    Input check passed
        Contains a 1 if the check of the input data passed and a 0 otherwise.
        If the check failed for one frequency, the following frequencies might
        be affected as well.
    Converged
        Contains a 1 if the simulation converged and a 0 otherwise. If the
        simulation did not converge, the relative error might be high.
    Num. iterations
        The number of iterations that were required to converge
    relative error
        The relative error of the final simulation
    Comp. time total
        The total computation time in seconds
    Comp. time assembling
        The computation time for assembling the matrices in seconds
    Comp. time solving
        The computation time for solving the matrices in seconds
    Comp. time post-proc
        The computation time for post-processing the results in seconds


    Parameters
    ----------
    folder : str, optional
        The path of the Mesh2HRTF project folder, i.e., the folder containing
        the subfolders EvaluationsGrids, NumCalc, and ObjectMeshes. The
        default, ``None`` uses the current working directory.

    Returns
    -------
    found_issues : bool
        ``True`` if issues were found, ``False`` otherwise
    report : str
        The report or an empty string if no issues were found
    """

    # get sources and number of sources and frequencies
    sources = glob.glob(os.path.join(folder, "NumCalc", "source_*"))
    num_sources = len(sources)

    if os.path.exists(os.path.join(folder, '..', "parameters.json")):
        with open(os.path.join(folder, '..', "parameters.json"), "r") as file:
            params = json.load(file)
    else:
        with open(os.path.join(folder, "parameters.json"), "r") as file:
            params = json.load(file)

    # sort source files (not read in correct order in some cases)
    nums = [int(source.split("_")[-1]) for source in sources]
    sources = np.array(sources)
    sources = sources[np.argsort(nums)]

    # parse all NC*.out files for all sources
    all_files, fundamentals, out, out_names = _parse_nc_out_files(
        sources, num_sources, params["num_frequencies"])

    # write report as csv file
    _write_project_reports(folder, all_files, out, out_names)

    # look for errors
    report = _check_project_report(folder, fundamentals, out)

    found_issues = True if report else False

    return found_issues, report


def _read_nodes_and_elements(folder, objects=None):
    """
    Read the nodes and elements of the evaluation grids or object meshes.

    Parameters
    ----------
    folder : str
        Folder containing the object. Must end with EvaluationGrids or
        Object Meshes
    objects : str, options
        Name of the object. The default ``None`` reads all objects in folder

    Returns
    -------
    grids : dict
        One item per object (with the item name being the object name). Each
        item has the sub-items `nodes`, `elements`, `num_nodes`, `num_elements`
    gridsNumNodes : int
        Number of nodes in all grids
    """
    # check input
    if os.path.basename(folder) not in ['EvaluationGrids', 'ObjectMeshes']:
        raise ValueError('folder must be EvaluationGrids or ObjectMeshes!')

    if objects is None:
        objects = os.listdir(folder)
        # discard hidden folders that might occur on Mac OS
        objects = [o for o in objects if not o.startswith('.')]
    elif isinstance(objects, str):
        objects = [objects]

    grids = {}
    gridsNumNodes = 0

    for grid in objects:
        tmpNodes = np.loadtxt(os.path.join(
            folder, grid, 'Nodes.txt'),
            delimiter=' ', skiprows=1, dtype=np.float64)

        tmpElements = np.loadtxt(os.path.join(
            folder, grid, 'Elements.txt'),
            delimiter=' ', skiprows=1, dtype=np.float64)

        grids[grid] = {
            "nodes": tmpNodes,
            "elements": tmpElements,
            "num_nodes": tmpNodes.shape[0],
            "num_elements": tmpElements.shape[0]}

        gridsNumNodes += grids[grid]['num_nodes']

    return grids, gridsNumNodes


def _read_numcalc_data(sources_num, num_frequencies, folder, data):
    """Read the sound pressure on the object meshes or evaluation grid."""
    pressure = []

    if data not in ['pBoundary', 'pEvalGrid', 'vBoundary', 'vEvalGrid']:
        raise ValueError(
            'data must be pBoundary, pEvalGrid, vBoundary, or vEvalGrid')

    for source in range(sources_num):

        tmpFilename = os.path.join(
            folder, 'NumCalc', f'source_{source+1}', 'be.out')
        tmpPressure, indices = _load_results(
            tmpFilename, data, num_frequencies)

        pressure.append(tmpPressure)

    pressure = np.transpose(np.array(pressure), (2, 0, 1))

    return pressure, indices


def _load_results(foldername, filename, num_frequencies):
    """
    Load results of the BEM calculation.

    Parameters
    ----------
    foldername : string
        The folder from which the data is loaded. The data to be read is
        located in the folder be.out inside NumCalc/source_*
    filename : string
        The kind of data that is loaded

        pBoundary
            The sound pressure on the object mesh
        vBoundary
            The sound velocity on the object mesh
        pEvalGrid
            The sound pressure on the evaluation grid
        vEvalGrid
            The sound velocity on the evaluation grid
    num_frequencies : int
        the number of simulated frequencies

    Returns
    -------
    data : numpy array
        Pressure or abs velocity values of shape (num_frequencies, numEntries)
    """

    # ---------------------check number of header and data lines---------------
    current_file = os.path.join(foldername, 'be.1', filename)
    numDatalines = None
    with open(current_file) as file:
        line = csv.reader(file, delimiter=' ', skipinitialspace=True)
        for _idx, li in enumerate(line):
            # read number of data points and head lines
            if len(li) == 2 and not li[0].startswith("Mesh"):
                numDatalines = int(li[1])

            # read starting index
            elif numDatalines and len(li) > 2:
                start_index = int(li[0])
                break
    if numDatalines is None:
        raise ValueError(
            f'{current_file} is empty!')
    # ------------------------------load data----------------------------------
    dtype = complex if filename.startswith("p") else float
    data = np.zeros((num_frequencies, numDatalines), dtype=dtype)

    for ii in range(num_frequencies):
        tmpData = []
        current_file = os.path.join(foldername, 'be.%d' % (ii+1), filename)
        with open(current_file) as file:

            line = csv.reader(file, delimiter=' ', skipinitialspace=True)

            for li in line:

                # data lines have 3 ore more entries
                if len(li) < 3 or li[0].startswith("Mesh"):
                    continue

                if filename.startswith("p"):
                    tmpData.append(complex(float(li[1]), float(li[2])))
                elif filename == "vBoundary":
                    tmpData.append(np.abs(complex(float(li[1]), float(li[2]))))
                elif filename == "vEvalGrid":
                    tmpData.append(np.sqrt(
                        np.abs(complex(float(li[1]), float(li[2])))**2 +
                        np.abs(complex(float(li[3]), float(li[4])))**2 +
                        np.abs(complex(float(li[5]), float(li[6])))**2))

        data[ii, :] = tmpData if tmpData else np.nan

    return data, np.arange(start_index, numDatalines + start_index)


def _check_project_report(folder, fundamentals, out):

    # return if there are no fundamental errors or other issues
    if not all(all(f) for f in fundamentals) and not np.any(out == -1) \
            and np.all(out[:, 3:5]):
        return

    # report detailed errors
    report = ""

    for ss in range(out.shape[2]):

        # currently we detect frequencies that were not calculated and
        # frequencies with convergence issues
        missing = "Frequency steps that were not calculated:\n"
        input_test = "Frequency steps with bad input:\n"
        convergence = "Frequency steps that did not converge:\n"

        any_missing = False
        any_input_failed = False
        any_convergence = False

        # loop frequencies
        for ff in range(out.shape[0]):

            f = out[ff, :, ss]

            # no value for frequency
            if f[1] == -1:
                any_missing = True
                missing += f"{int(f[0])}, "
                continue

            # input data failed
            if f[3] == 0:
                any_input_failed = True
                input_test += f"{int(f[0])}, "

            # convergence value is zero
            if f[4] == 0:
                any_convergence = True
                convergence += f"{int(f[0])}, "

        if any_missing or any_input_failed or any_convergence:
            report += f"Detected issues for source {ss+1}\n"
            report += "----------------------------\n"
            if any_missing:
                report += missing[:-2] + "\n\n"
            if any_input_failed:
                report += input_test[:-2] + "\n\n"
            if any_convergence:
                report += convergence[:-2] + "\n\n"

    if not report:
        report = ("\nDetected unknown issues\n"
                  "-----------------------\n"
                  "Check the project reports in Output2HRTF,\n"
                  "and the NC*.out files in NumCalc/source_*\n\n")

    report += ("For more information check Output2HRTF/report_source_*.csv "
               "and the NC*.out files located at NumCalc/source_*")

    # write to disk
    report_name = os.path.join(
        folder, "Output2HRTF", "report_issues.txt")
    with open(report_name, "w") as f_id:
        f_id.write(report)

    return report


def _write_project_reports(folder, all_files, out, out_names):
    """
    Write project report to disk at folder/Output2HRTF/report_source_*.csv.

    For description of input parameter refer to write_output_report and
    _parse_nc_out_files
    """

    # loop sources
    for ss in range(out.shape[2]):

        report = ", ".join(out_names) + "\n"

        # loop frequencies
        for ff in range(out.shape[0]):
            f = out[ff, :, ss]
            report += (
                f"{int(f[0])}, "                # frequency step
                f"{float(f[1])}, "              # frequency in Hz
                f"{all_files[ss][int(f[2])]},"  # NC*.out file
                f"{int(f[3])}, "                # input check
                f"{int(f[4])}, "                # convergence
                f"{int(f[5])}, "                # number of iterations
                f"{float(f[6])}, "              # relative error
                f"{int(f[7])}, "                # total computation time
                f"{int(f[8])}, "                # assembling equations time
                f"{int(f[9])}, "                # solving equations time
                f"{int(f[10])}\n"               # post-processing time
                )

        # write to disk
        report_name = os.path.join(
            folder, "Output2HRTF", f"report_source_{ss + 1}.csv")
        with open(report_name, "w") as f_id:
            f_id.write(report)


def _parse_nc_out_files(sources, num_sources, num_frequencies):
    """
    Parse all NC*.out files for all sources.

    This function should never raise a value error, regardless of how mess
    NC*.out files are. Looking for error is done at a later step.

    Parameters
    ----------
    sources : list of strings
        full path to the source folders
    num_sources : int
        number of sources - len(num_sources)
    num_frequencies : int
        number of frequency steps

    Returns
    -------
    out : numpy array
        containing the extracted information for each frequency step
    out_names : list of string
        verbal information about the columns of `out`
    """

    # array for reporting fundamental errors
    fundamentals = []
    all_files = []

    # array for saving the detailed report
    out_names = ["frequency step",         # 0
                 "frequency in Hz",        # 1
                 "NC input file",          # 2
                 "Input check passed",     # 3
                 "Converged",              # 4
                 "Num. iterations",        # 5
                 "relative error",         # 6
                 "Comp. time total",       # 7
                 "Comp. time assembling",  # 8
                 "Comp. time solving",     # 9
                 "Comp. time post-proc."]  # 10
    out = -np.ones((num_frequencies, 11, num_sources))
    # values for steps
    out[:, 0] = np.arange(1, num_frequencies + 1)[..., np.newaxis]
    # values indicating failed input check and non-convergence
    out[:, 3] = 0
    out[:, 4] = 0

    # regular expression for finding a number that can be int or float
    re_number = r"(\d+(?:\.\d+)?)"

    # loop sources
    for ss, source in enumerate(sources):

        # list of NC*.out files for parsing
        files = glob.glob(os.path.join(source, "NC*.out"))

        # make sure that NC.out is first
        nc_out = os.path.join(source, "NC.out")
        if nc_out in files and files.index(nc_out):
            files = [files.pop(files.index(nc_out))] + files

        # update fundamentals
        fundamentals.append([0 for f in range(len(files))])
        all_files.append([os.path.basename(f) for f in files])

        # get content from all NC*.out
        for ff, file in enumerate(files):

            # read the file and join all lines
            with open(file, "r") as f_id:
                lines = f_id.readlines()
            lines = "".join(lines)

            # split header and steps
            lines = lines.split(
                ">> S T E P   N U M B E R   A N D   F R E Q U E N C Y <<")

            # look for fundamental errors
            if len(lines) == 1:
                fundamentals[ss][ff] = 1
                continue

            # parse frequencies (skip header)
            for line in lines[1:]:

                # find frequency step
                idx = re.search(r'Step \d+,', line)
                if idx:
                    step = int(line[idx.start()+5:idx.end()-1])

                # write number of input file (replaced by string later)
                out[step-1, 2, ss] = ff

                # find frequency
                idx = re.search(f'Frequency = {re_number} Hz', line)
                if idx:
                    out[step-1, 1, ss] = float(
                        line[idx.start()+12:idx.end()-3])

                # check if the input data was ok
                if "Too many integral points in the theta" not in line:
                    out[step-1, 3, ss] = 1

                # check and write convergence
                if 'Maximum number of iterations is reached!' not in line:
                    out[step-1, 4, ss] = 1

                # check iterations
                idx = re.search(r'number of iterations = \d+,', line)
                if idx:
                    out[step-1, 5, ss] = int(line[idx.start()+23:idx.end()-1])

                # check relative error
                idx = re.search('relative error = .+', line)
                if idx:
                    out[step-1, 6, ss] = float(line[idx.start()+17:idx.end()])

                # check time stats
                # -- assembling
                idx = re.search(
                    r'Assembling the equation system         : \d+',
                    line)
                if idx:
                    out[step-1, 8, ss] = float(line[idx.start()+41:idx.end()])

                # -- solving
                idx = re.search(
                    r'Solving the equation system            : \d+',
                    line)
                if idx:
                    out[step-1, 9, ss] = float(line[idx.start()+41:idx.end()])

                # -- post-pro
                idx = re.search(
                    r'Post processing                        : \d+',
                    line)
                if idx:
                    out[step-1, 10, ss] = float(line[idx.start()+41:idx.end()])

                # -- total
                idx = re.search(
                    r'Total                                  : \d+',
                    line)
                if idx:
                    out[step-1, 7, ss] = float(line[idx.start()+41:idx.end()])

    return all_files, fundamentals, out, out_names


def read_evaluation_grid(name):
    """
    Read Mesh2HRTF evaluation grid.

    Parameters
    ----------
    name : str
        Name of the folder containing the nodes of the evaluation grid in
        Nodes.txt
    show : bool, optional
        If ``True`` the points of the evaluation grid are plotted. The default
        is ``False``.

    Returns
    -------
    coordinates : pyfar Coordinates
        The points of the evaluation grid as a pyfar Coordinates object
    """

    # check if the grid exists
    if not os.path.isfile(os.path.join(name, "Nodes.txt")):
        raise ValueError(f"{os.path.join(name, 'Nodes.txt')} does not exist")

    # read the nodes
    with open(os.path.join(name, "Nodes.txt"), "r") as f_id:
        nodes = f_id.readlines()

    # get number of nodes
    N = int(nodes[0].strip())
    points = np.zeros((N, 3))

    # get points (first entry is node number)
    for nn in range(N):
        node = nodes[nn+1].strip().split(" ")
        points[nn, 0] = float(node[1])
        points[nn, 1] = float(node[2])
        points[nn, 2] = float(node[3])

    # make coordinates object
    coordinates = pf.Coordinates(points[:, 0], points[:, 1], points[:, 2])

    return coordinates

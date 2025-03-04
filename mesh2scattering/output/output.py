"""Write output for NumCalc."""

import os
import warnings
import json
import numpy as np
import pyfar as pf
import glob
import sofar as sf
import csv
import re


def write_pressure(folder):
    """
    Process NumCalc output and write data to disk.

    Processing the data is done in the following steps

    1. Read project parameter from `parameters.json`
    2. use :py:func:`~write_output_report` to parse files in
       project_folder/NumCalc/source_*/NC*.out, write project report to
       project_folder/report/report_source_*.csv. Raise a warning if any
       issues were detected and write report_issues.txt to the same folder
    3. Read simulated pressures from project_folder/NumCalc/source_*/be.out.
       This and the following steps are done, even if an issue was detected in
       the previous step
    4. save the sound pressure for each evaluation grid to
       save the a SOFA file

    Parameters
    ----------
    folder : str, optional
        The path of the NumCalc project folder, i.e., the folder containing
        the subfolders EvaluationsGrids, NumCalc, and ObjectMeshes.

    Returns
    -------
    written_files : list of str
        List of the full path of the written SOFA files.

    """
    if not os.path.exists(folder):
        raise ValueError(
            'Folder need to exists.')

    # write the project report and check for issues
    print('\n Writing the project report ...')
    found_issues, report = write_output_report(folder)

    if found_issues:
        warnings.warn(report, stacklevel=2)

    # read sample data
    evaluationGrids, params = _read_numcalc(
        folder)

    written_files = []
    # process BEM data for writing HRTFs and HRIRs to SOFA files
    for i_grid, grid in enumerate(evaluationGrids):

        # get pressure as SOFA object (all following steps are run on SOFA
        # objects. This way they are available to other users as well)
        source_position = np.array(params['sources'])
        source_type = params['source_type']
        if source_type == 'Plane wave':
            source_position = -source_position*99
        receiver_position = np.array(evaluationGrids[grid]['nodes'][:, 1:4])

        # apply symmetry of reference sample
        data = pf.FrequencyData(
            evaluationGrids[grid]['pressure'], params['frequencies'])
        receiver_coords = _cart_coordinates(receiver_position)
        source_coords = _cart_coordinates(source_position)
        source_coords.weights = params['sources_weights']
        receiver_coords.weights = np.array(
            params['evaluation_grids_weights'][i_grid])

        # write data
        c = float(params['speed_of_sound'])
        Lbyl = params['structural_wavelength'] / c * data.frequencies
        sofa = _create_pressure_sofa(
            data,
            Lbyl,
            source_coords,
            receiver_coords,
            structural_wavelength=params['structural_wavelength'],
            structural_wavelength_x=params['structural_wavelength_x'],
            structural_wavelength_y=params['structural_wavelength_y'],
            sample_diameter=params['sample_diameter'],
            speed_of_sound=params['speed_of_sound'],
            density_of_medium=params['density_of_medium'],
            mesh2scattering_version=params['mesh2scattering_version'],
            model_scale=params['model_scale'],
            symmetry_azimuth=params['symmetry_azimuth'],
            symmetry_rotational=params['symmetry_rotational'],
            )

        sofa.GLOBAL_Title = folder.split(os.sep)[-1]

        file_path = os.path.join(
            folder, '..',
            f'{params["project_title"]}_{grid}.pressure.sofa')
        # write scattered sound pressure to SOFA file
        sf.write_sofa(file_path, sofa)
        written_files.append(file_path)
    return written_files


def _create_pressure_sofa(
        data, Lbyl, sources, receivers, structural_wavelength,
        structural_wavelength_x, structural_wavelength_y,
        sample_diameter, speed_of_sound,
        density_of_medium,mesh2scattering_version,
        model_scale=1, symmetry_azimuth=None,
        symmetry_rotational=False,
        ):
    """Write complex pressure data to SOFA object from NumCalc simulation.

    Parameters
    ----------
    data : :py:class:`~pyfar.classes.audio.FrequencyData`
        the complex pressure data.
    Lbyl : numpy.ndarray
        structural wavelength over wavelength of sound.
    sources : :py:class:`~pyfar.classes.coordinates.Coordinates`
        source positions in cartesian coordinates.
    receivers : :py:class:`~pyfar.classes.coordinates.Coordinates`
        Receivers positions in cartesian coordinates.
    structural_wavelength : float
        structural wavelength in main direction (x) in meters.
    structural_wavelength_x : float
        structural wavelength in x direction in meters.
    structural_wavelength_y : float
        structural wavelength in y direction in meters.
    sample_diameter : float
        diameter of the sample in meters.
    speed_of_sound : float
        speed of sound in m/s
    density_of_medium : float
        density of the medium in kg/m^3.
    mesh2scattering_version : str
        version string of mesh2scattering
    model_scale : int, optional
        scale of the surface, by default 1.
    symmetry_azimuth : list, optional
        the azimuth angles in degree, where the symmetry axes are,
        by default [].
    symmetry_rotational : bool, optional
        Whether the surface is rotational symmetric, by default False

    Returns
    -------
    sf.Sofa
        sofa file.
    """
    # create empty SOFA object
    convention = 'GeneralTF' if type(
        data) is pf.FrequencyData else 'GeneralFIR'

    assert speed_of_sound is not None
    assert speed_of_sound > 250
    assert speed_of_sound < 350

    sofa = sf.Sofa(convention)

    # write meta data
    sofa.GLOBAL_ApplicationName = 'Mesh2scattering'
    sofa.GLOBAL_ApplicationVersion = mesh2scattering_version
    sofa.GLOBAL_History = 'numerically simulated data'

    # Source and receiver data
    sofa.EmitterPosition = sources.cartesian
    sofa.EmitterPosition_Units = 'meter'
    sofa.EmitterPosition_Type = 'cartesian'

    sources_sph = sources.spherical_elevation
    sources_sph = pf.rad2deg(sources_sph)
    sofa.SourcePosition = sources_sph
    sofa.SourcePosition_Units = 'degree, degree, metre'
    sofa.SourcePosition_Type = 'spherical'

    sofa.ReceiverPosition = receivers.cartesian
    sofa.ReceiverPosition_Units = 'meter'
    sofa.ReceiverPosition_Type = 'cartesian'

    if type(data) is pf.FrequencyData:
        Lbyl = np.array(Lbyl)
        if structural_wavelength == 0:
            f = Lbyl
        else:
            f = Lbyl/structural_wavelength*speed_of_sound
        sofa.N = data.frequencies
        sofa.add_variable(
            'OriginalFrequencies', f, 'double', 'N')
        sofa.add_variable(
            'RealScaleFrequencies', f/model_scale, 'double', 'N')
        sofa.add_variable(
            'Lbyl', Lbyl, 'double', 'N')
        # HRTF/HRIR data
        if data.cshape[0] != sources.csize:
            data.freq = np.swapaxes(data.freq, 0, 1)
        sofa.Data_Real = np.real(data.freq)
        sofa.Data_Imag = np.imag(data.freq)
    else:
        sofa.Data_IR = data.time
        sofa.Data_SamplingRate = data.sampling_rate
        sofa.Data_Delay = np.zeros((1, receivers.csize))


    sofa.add_variable(
        'SampleStructuralWavelength', structural_wavelength, 'double', 'I')
    sofa.add_variable(
        'SampleStructuralWavelengthX', structural_wavelength_x, 'double', 'I')
    sofa.add_variable(
        'SampleStructuralWavelengthY', structural_wavelength_y, 'double', 'I')
    sofa.add_variable(
        'SampleModelScale', model_scale, 'double', 'I')
    sofa.add_variable(
        'SampleDiameter', sample_diameter, 'double', 'I')
    sofa.add_variable(
        'SpeedOfSound', speed_of_sound, 'double', 'I')
    sofa.add_variable(
        'DensityOfMedium', density_of_medium, 'double', 'I')
    sofa.add_variable(
        'ReceiverWeights', receivers.weights, 'double', 'R')
    sofa.add_variable(
        'SourceWeights', sources.weights, 'double', 'E')
    if symmetry_azimuth is None:
        symmetry_azimuth_str = ''
    else:
        symmetry_azimuth_str = ','.join([f'{a}' for a in symmetry_azimuth])
    sofa.add_variable(
        'SampleSymmetryAzimuth', symmetry_azimuth_str, 'string', 'S')
    sofa.add_variable(
        'SampleSymmetryRotational',
        1 if symmetry_rotational else 0, 'double', 'I')

    return sofa

def _cart_coordinates(xyz):
    return pf.Coordinates(xyz[:, 0], xyz[:, 1], xyz[:, 2])


def _read_numcalc(folder=None):
    """
    Process NumCalc output and write data to disk.

    Processing the data is done in the following steps

    1. Read project parameter `from parameters.json`
    2. use :py:func:`~write_output_report` to parse files in
       project_folder/NumCalc/source_*/NC*.out, write project report to
       project_folder/report/report_source_*.csv. Raise a warning if any
       issues were detected and write report_issues.txt to the same folder
    3. Read simulated pressures from project_folder/NumCalc/source_*/be.out.
       This and the following steps are done, even if an issue was detected in
       the previous step
    4. write the reflected sound pressure to sofa file.

    Parameters
    ----------
    folder : str, optional
        The path of the NumCalc project folder, i.e., the folder containing
        the subfolders EvaluationsGrids, NumCalc, and ObjectMeshes. The
        default, ``None`` uses the current working directory.
    """
    # check input
    if folder is None:
        folder = os.getcwd()

    # check and load parameters, required parameters are:
    # mesh2scattering_version, reference, computeHRIRs, speedOfSound,
    # densityOfAir,
    # sources_num, sourceType, sources, sourceArea,
    # num_frequencies, frequencies
    params = os.path.join(folder, 'parameters.json')
    if not os.path.isfile(params):
        raise ValueError((
            f"The folder {folder} is not a valid Mesh2scattering project. "
            "It must contain the file 'parameters.json'"))

    with open(params, "r") as file:
        params = json.load(file)

    # get source positions
    source_coords = pf.Coordinates()
    source_coords.cartesian = np.array(params['sources'])


    # get the evaluation grids
    evaluationGrids, _ = _read_nodes_and_elements(
        os.path.join(folder, 'EvaluationGrids'))

    # Load EvaluationGrid data
    num_sources = source_coords.csize

    if not len(evaluationGrids) == 0:
        pressure, _ = _read_numcalc_data(
            num_sources, len(params["frequencies"]),
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

    NumCalc (mesh2scattering's numerical core) writes information about the
    simulations to the files `NC*.out` located under `NumCalc/source_*`. The
    file `NC.out` exists if NumCalc was ran without the additional command line
    parameters ``-istart`` and ``-iend``. If these parameters were used, there
    is at least one `NC\*-\*.out`. If this is the case, information from
    `NC\*-\*.out` overwrites information from NC.out in the project report.

    .. note::

        The project reports are written to the files
        `report/report_source_*.csv`. If issues were detected, they are
        listed in `report/report_issues.csv`.

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
        The path of the NumCalc project folder, i.e., the folder containing
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

    if folder is None:
        folder = os.getcwd()

    # get sources and number of sources and frequencies
    sources = glob.glob(os.path.join(folder, "NumCalc", "source_*"))
    num_sources = len(sources)

    with open(os.path.join(folder, "parameters.json"), "r") as file:
        params = json.load(file)

    # sort source files (not read in correct order in some cases)
    nums = [int(source.split("_")[-1]) for source in sources]
    sources = np.array(sources)
    sources = sources[np.argsort(nums)]

    # parse all NC*.out files for all sources
    all_files, fundamentals, out, out_names = _parse_nc_out_files(
        sources, num_sources, len(params["frequencies"]))

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


def _read_numcalc_data(n_sources, n_frequencies, folder, data):
    """Read the sound pressure on the object meshes or evaluation grid."""
    pressure = []

    if data not in ['pBoundary', 'pEvalGrid', 'vBoundary', 'vEvalGrid']:
        raise ValueError(
            'data must be pBoundary, pEvalGrid, vBoundary, or vEvalGrid')

    for source in range(n_sources):

        tmpFilename = os.path.join(
            folder, 'NumCalc', f'source_{source+1}', 'be.out')
        tmpPressure, indices = _load_results(
            tmpFilename, data, n_frequencies)

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
            if len(li) == 2 and not (
                    li[0].startswith("mesh2scattering") or li[0].startswith(
                        "Mesh2HRTF")):
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
                  "Check the project reports in report,\n"
                  "and the NC*.out files in NumCalc/source_*\n\n")

    report += ("For more information check report/report_source_*.csv "
               "and the NC*.out files located at NumCalc/source_*")

    # write to disk
    report_name = os.path.join(
        folder, "report", "report_issues.txt")
    with open(report_name, "w") as f_id:
        f_id.write(report)

    return report


def _write_project_reports(folder, all_files, out, out_names):
    """
    Write project report to disk at folder/report/report_source_*.csv.

    For description of input parameter refer to write_output_report and
    _parse_nc_out_files
    """
    if not os.path.exists(os.path.join(folder, 'report')):
        os.mkdir(os.path.join(folder, 'report'))

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
            folder, "report", f"report_source_{ss + 1}.csv")
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
    Read NumCalc evaluation grid.

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

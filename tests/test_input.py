import mesh2scattering as m2s
import os
import numpy as np
import pyfar as pf
import trimesh
import filecmp
import json
import numpy.testing as npt



def test_import():
    from mesh2scattering import input  # noqa: A004
    assert input


def test_write_project(tmpdir, simple_mesh):
    project_path = os.path.join(tmpdir, 'project')
    project_title = 'test_project'
    frequencies = np.array([500])
    sound_sources = m2s.input.SoundSource(
        pf.Coordinates(1, 0, 1, weights=1),
        m2s.input.SoundSourceType.POINT_SOURCE)
    points = pf.samplings.sph_lebedev(sh_order=10)
    evaluation_grid = m2s.input.EvaluationGrid.from_spherical(
        points,
        'example_grid')
    surface_description = m2s.input.SurfaceDescription()
    sample_mesh = m2s.input.SampleMesh(
        simple_mesh,
        surface_description,
        0.8,
        m2s.input.SampleShape.ROUND,
    )
    bem_method = 'ML-FMM BEM'
    m2s.input.write_scattering_project_numcalc(
        project_path,
        project_title,
        frequencies,
        sound_sources,
        [evaluation_grid],
        sample_mesh,
        bem_method,
        speed_of_sound=346.18,
        density_of_medium=1.1839,
        )

    # check create project folders
    assert os.path.isdir(os.path.join(
        project_path, 'sample'))
    assert os.path.isdir(os.path.join(
        project_path, 'sample', 'ObjectMeshes'))
    assert os.path.isdir(os.path.join(
        project_path, 'sample', 'EvaluationGrids'))
    assert os.path.isdir(os.path.join(
        project_path, 'sample', 'NumCalc'))

    # check mesh files
    assert os.path.isfile(os.path.join(
        project_path, 'sample', 'ObjectMeshes', 'Reference.stl'))
    assert os.path.isfile(os.path.join(
        project_path, 'sample', 'ObjectMeshes', 'Reference', 'Nodes.txt'))
    assert os.path.isfile(os.path.join(
        project_path, 'sample', 'ObjectMeshes', 'Reference', 'Elements.txt'))

    # check evaluation grid files
    assert os.path.isfile(os.path.join(
        project_path, 'sample', 'EvaluationGrids',
        evaluation_grid.name, 'Nodes.txt'))
    assert os.path.isfile(os.path.join(
        project_path, 'sample', 'EvaluationGrids',
        evaluation_grid.name, 'Elements.txt'))

    # check numcalc files
    assert os.path.isfile(os.path.join(
        project_path, 'sample', 'NumCalc', 'source_1', 'NC.inp'))
    assert not os.path.isfile(os.path.join(
        project_path, 'sample', 'NumCalc', 'source_2', 'NC.inp'))

    # check parameters.json
    json_path = os.path.join(
        project_path, 'parameters.json')
    assert os.path.isfile(json_path)

    with open(json_path, 'r') as file:
        params = json.load(file)

    assert params['project_title'] == project_title
    assert params['bem_method'] == bem_method

    assert params['speed_of_sound'] == 346.18
    assert params['density_of_medium'] == 1.1839
    surf = surface_description
    assert params['structural_wavelength'] == surf.structural_wavelength_x
    assert params['structural_wavelength_x'] == surf.structural_wavelength_x
    assert params['structural_wavelength_y'] == surf.structural_wavelength_y
    assert params['model_scale'] == surf.model_scale
    assert params['symmetry_azimuth'] == surf.symmetry_azimuth
    assert params['symmetry_rotational'] == surf.symmetry_rotational
    assert params['sample_diameter'] == sample_mesh.sample_diameter

    npt.assert_array_almost_equal(params['frequencies'], frequencies)

    assert params['source_type'] == 'Point source'
    npt.assert_array_almost_equal(
        params['sources'], sound_sources.source_coordinates.cartesian)
    npt.assert_array_almost_equal(
        params['sources_weights'], sound_sources.source_coordinates.weights)

    evaluation_grids = [
        np.array(r) for r in params['evaluation_grids_coordinates']]
    npt.assert_array_almost_equal(
        evaluation_grids[0], points.cartesian)
    evaluation_grids_weights = np.array(params['evaluation_grids_weights'])
    npt.assert_array_almost_equal(
        evaluation_grids_weights[0], points.weights)
    assert evaluation_grids_weights[0].shape[0] == evaluation_grids[0].shape[0]


def test__write_nc_inp(tmpdir):
    # test write nc inp file vs online documentation
    # https://github.com/Any2HRTF/Mesh2HRTF/wiki/Structure_of_NC.inp
    version = '0.1.0'
    project_path = os.path.join(tmpdir, 'project')
    project_title = 'test_project'
    speed_of_sound = 346.18
    density_of_medium = 1.1839
    frequencies = np.array([500])
    evaluation_grid_names = ['example_grid']
    source_type = m2s.input.SoundSourceType.POINT_SOURCE
    source_positions = pf.Coordinates(1, 0, 1, weights=1)
    n_mesh_elements = 100
    n_mesh_nodes = 50
    n_grid_elements = 200
    n_grid_nodes = 70
    method = 'ML-FMM BEM'
    os.mkdir(project_path)
    os.mkdir(os.path.join(project_path, 'NumCalc'))
    m2s.input.input._write_nc_inp(
        project_path, version, project_title,
        speed_of_sound, density_of_medium,
        frequencies,
        evaluation_grid_names,
        source_type, source_positions,
        n_mesh_elements, n_mesh_nodes, n_grid_elements, n_grid_nodes, method)

    # test if files are there
    assert os.path.isdir(os.path.join(project_path, 'NumCalc', 'source_1'))
    assert not os.path.isdir(os.path.join(project_path, 'NumCalc', 'source_2'))
    assert os.path.isfile(os.path.join(
        project_path, 'NumCalc', 'source_1', 'NC.inp'))
    assert not os.path.isfile(os.path.join(
        project_path, 'NumCalc', 'source_2', 'NC.inp'))

    # test if the file is correct
    n_bins = frequencies.size
    NC_path = os.path.join(project_path, 'NumCalc', 'source_1', 'NC.inp')
    with open(NC_path, 'r') as f:
        content = "".join(f.readlines())

    # Controlparameter I [%d]
    assert (
        '## Controlparameter I\n'
        '0\n') in content

    # Controlparameter II [%d %d %f %f]
    assert (
        '## Controlparameter II\n'
        f'1 {n_bins} 0.000001 0.00e+00 1 0 0\n') in content

    # check Frequency Curve
    frequency_curve = (
        '## Load Frequency Curve \n'
        f'0 {n_bins+1}\n'
        '0.000000 0.000000e+00 0.0\n')
    frequency_curve += ''.join([
        f'0.{i+1:06d} {(frequencies[i]/10000):04f}e+04 0.0\n' for i in range(n_bins)])
    assert frequency_curve in content

    # Main Parameters I [%d %d %d %d %d %d %d %d]
    bem_method_id = 0 if method == 'BEM' else 1 if method == 'SL-FMM BEM' else 4
    n_nodes = n_mesh_nodes + n_grid_nodes
    n_elements = n_mesh_elements + n_grid_elements
    main_1 = (
        '## 1. Main Parameters I\n'
        f'2 {n_elements} {n_nodes} 0 0 2 1 {bem_method_id} 0\n')
    assert main_1 in content

    # MainParameters II [%d %d %d %d %d]
    n_plane_waves = source_positions.csize if (
        source_type == m2s.input.SoundSourceType.PLANE_WAVE) else 0
    n_point_sources = source_positions.csize if (
        source_type == m2s.input.SoundSourceType.POINT_SOURCE) else 0
    main_2 = (
        '## 2. Main Parameters II\n'
        f'{n_plane_waves} {n_point_sources} 0 0.0000e+00 0 0\n')
    assert main_2 in content

    #Main Parameters III [%d]
    main_3 = (
        '## 3. Main Parameters III\n'
        '0\n')
    assert main_3 in content

    #Main Parameters IV [%f %f %f]
    main_4 = (
        '## 4. Main Parameters IV\n'
        f'{speed_of_sound} {density_of_medium} 1.0\n')
    assert main_4 in content

    #NODES [%s ... %s]
    nodes = (
        'NODES\n'
        '../../ObjectMeshes/Reference/Nodes.txt\n')
    nodes += ''.join([
        f'../../EvaluationGrids/{grid}/Nodes.txt\n' \
            for grid in evaluation_grid_names])
    assert nodes in content

    #ELEMENTS [%s ... %s]
    elements = (
        'ELEMENTS\n'
        '../../ObjectMeshes/Reference/Elements.txt\n')
    elements += ''.join([
        f'../../EvaluationGrids/{grid}/Elements.txt\n' \
            for grid in evaluation_grid_names])
    assert elements in content

    #SYMMETRY [%d %d %d %f %f %f]
    symmetry = (
        '# SYMMETRY\n'
        '# 0 0 0\n'
        '# 0.0000e+00 0.0000e+00 0.0000e+00\n')
    assert symmetry in content

    #BOUNDARY : ELEM [%d] TO [%d %s %f %d %f %d]
    boundary = '##\nBOUNDARY\nRETU\n'
    assert boundary in content

    #PLANE WAVES [%d %f %f %f %d %f %d %f]
    xx = source_positions.x
    yy = source_positions.y
    zz = source_positions.z
    if n_plane_waves:
        plane_waves = (
            'PLANE WAVES\n')
        plane_waves += ''.join([
            f'{i} {xx[i]} {yy[i]} {zz[i]} 0.1 -1 0.0 -1\n' \
                for i in range(source_positions.csize)])
        assert plane_waves in content

    if n_point_sources:
        point_sources = (
            'POINT SOURCES\n')
        point_sources += ''.join([
            f'{i} {xx[i]} {yy[i]} {zz[i]} 0.1 -1 0.0 -1\n' \
                for i in range(source_positions.csize)])
        assert point_sources in content

    # no curves in this case
    assert '\n# CURVES\n' in content
    assert '\nPOST PROCESS\n' in content
    assert '\nEND\n' in content


# @pytest.mark.parametrize("n_dim", [3, 2])
# @pytest.mark.parametrize(('coordinates'), [True])
# def test_read_and_write_evaluation_grid(n_dim, coordinates):
#     cwd = os.path.dirname(__file__)
#     data_grids = os.path.join(cwd, 'resources', 'evaluation_grids')

#     tmp = TemporaryDirectory()

#     # sampling grids
#     if n_dim == 3:
#         # 3D sampling grid (Lebedev, first order)
#         points = np.array([
#             [1., 0., 0.],
#             [-1., 0., 0.],
#             [0, 1., 0.],
#             [0, -1., 0.],
#             [0, 0., 1.],
#             [0, 0., -1.]])
#         discard = None
#     else:
#         # 2D sampling grid (all z = 0)
#         points = np.array([
#             [1., 0., 0.],
#             [-1., 0., 0.],
#             [0, 1., 0.],
#             [0, -1., 0.]])
#         discard = "z"

#     # pass as Coordinates object
#     if coordinates:
#         points = pf.Coordinates(points[:, 0], points[:, 1], points[:, 2])

#     # write grid
#     m2s.input.write_evaluation_grid(
#         points, os.path.join(tmp.name, "test"), discard=discard)

#     # check the nodes and elements
#     for file in ["Nodes.txt", "Elements.txt"]:
#         with open(os.path.join(data_grids, f"{n_dim}D", file), "r") as f:
#             ref = "".join(f.readlines())
#         with open(os.path.join(tmp.name, "test", file), "r") as f:
#             test = "".join(f.readlines())

#         assert test == ref

#     # read the grid
#     coordinates = m2s.output.read_evaluation_grid(
#         os.path.join(tmp.name, "test"))

#     # check grid
#     assert isinstance(coordinates, pf.Coordinates)
#     npt.assert_equal(coordinates.get_cart(), points)




# def test_write_scattering_parameter(tmpdir):
#     sourcePoints = pf.samplings.sph_equal_angle(10, 10)
#     sourcePoints = sourcePoints[sourcePoints.elevation >= 0]
#     sourcePoints = sourcePoints[sourcePoints.azimuth <= np.pi/2]

#     frequencies = pf.dsp.filter.fractional_octave_frequencies(
#         3, (500, 5000))[0]
#     path = os.path.join(
#         m2s.utils.program_root(), '..',
#         'tests', 'resources', 'mesh', 'sine_5k')
#     sample_path = os.path.join(path, 'sample.stl')
#     reference_path = os.path.join(path, 'reference.stl')
#     receiver_delta_deg = 1
#     receiver_radius = 5

#     structural_wavelength = 0
#     sample_diameter = 0.8
#     model_scale = 2.5
#     symmetry_azimuth = [90, 180]
#     symmetry_rotational = False

#     receiverPoints = pf.samplings.sph_equal_angle(
#         receiver_delta_deg, receiver_radius)
#     receiverPoints = receiverPoints[receiverPoints.get_sph()[..., 1] < np.pi/2]

#     # execute
#     m2s.input.write_scattering_project(
#         project_path=tmpdir,
#         frequencies=frequencies,
#         sample_path=sample_path,
#         reference_path=reference_path,
#         receiver_coords=receiverPoints,
#         source_coords=sourcePoints,
#         structural_wavelength=structural_wavelength,
#         model_scale=model_scale,
#         sample_diameter=sample_diameter,
#         symmetry_azimuth=symmetry_azimuth,
#         symmetry_rotational=symmetry_rotational,
#         )

#     # test parameters
#     f = open(os.path.join(tmpdir, 'parameters.json'))
#     paras = json.load(f)
#     source_list = [list(i) for i in list(sourcePoints.get_cart())]
#     receiver_list = [list(i) for i in list(receiverPoints.get_cart())]
#     parameters = {
#         # project Info
#         "project_title": 'scattering pattern',
#         "mesh2scattering_path": m2s.utils.program_root(),
#         "mesh2scattering_version": m2s.__version__,
#         "bem_version": 'ML-FMM BEM',
#         # Constants
#         "speed_of_sound": float(346.18),
#         "density_of_medium": float(1.1839),
#         # Sample Information, post processing
#         "structural_wavelength": structural_wavelength,
#         "model_scale": model_scale,
#         "sample_diameter": sample_diameter,
#         # symmetry information
#         "symmetry_azimuth": symmetry_azimuth,
#         "symmetry_rotational": symmetry_rotational,
#         # frequencies
#         "num_frequencies": len(frequencies),
#         "min_frequency": frequencies[0],
#         "max_frequency": frequencies[-1],
#         "frequencies": list(frequencies),
#         # Source definition
#         "source_type": 'Point source',
#         "sources_num": len(source_list),
#         "sources": source_list,
#         # Receiver definition
#         "receivers_num": len(receiver_list),
#         "receivers": receiver_list,
#     }
#     npt.assert_array_almost_equal(paras['receivers'], parameters['receivers'])
#     paras['receivers'] = parameters['receivers']
#     npt.assert_equal(paras, parameters)
#     # test folder structure
#     assert os.path.isdir(os.path.join(tmpdir, 'sample'))
#     assert os.path.isdir(os.path.join(tmpdir, 'reference'))
#     assert os.path.isdir(os.path.join(tmpdir, 'sample', 'EvaluationGrids'))
#     assert os.path.isdir(os.path.join(tmpdir, 'reference', 'EvaluationGrids'))
#     assert os.path.isdir(os.path.join(tmpdir, 'sample', 'NumCalc'))
#     assert os.path.isdir(os.path.join(tmpdir, 'reference', 'NumCalc'))
#     assert os.path.isdir(os.path.join(tmpdir, 'sample', 'ObjectMeshes'))
#     assert os.path.isdir(os.path.join(tmpdir, 'reference', 'ObjectMeshes'))
#     assert os.path.isfile(os.path.join(
#         tmpdir, 'sample', 'ObjectMeshes', 'sample.stl'))
#     assert os.path.isfile(os.path.join(
#         tmpdir, 'reference', 'ObjectMeshes', 'reference.stl'))

#     # test sources
#     for i in range(91):
#         assert os.path.isdir(
#             os.path.join(tmpdir, 'sample', 'NumCalc', f'source_{i+1}'))
#     assert not os.path.isdir(
#         os.path.join(tmpdir, 'sample', 'NumCalc', f'source_{92}'))
#     for i in range(10):
#         assert os.path.isdir(
#             os.path.join(tmpdir, 'reference', 'NumCalc', f'source_{i+1}'))
#     assert not os.path.isdir(
#         os.path.join(tmpdir, 'reference', 'NumCalc', f'source_{11}'))


# def test_write_scattering_parameter_one_source(tmpdir):
#     source_coords = pf.Coordinates(1, 0, 1)
#     frequencies = pf.dsp.filter.fractional_octave_frequencies(
#         3, (500, 5000))[0]
#     path = os.path.join(
#         m2s.utils.program_root(), '..',
#         'tests', 'resources', 'mesh', 'sine_5k')
#     sample_path = os.path.join(path, 'sample.stl')
#     reference_path = os.path.join(path, 'reference.stl')
#     receiver_delta_deg = 1
#     receiver_radius = 5

#     structural_wavelength = 0
#     sample_diameter = 0.8
#     model_scale = 2.5
#     symmetry_azimuth = [90, 180]
#     symmetry_rotational = False

#     receiverPoints = pf.samplings.sph_equal_angle(
#         receiver_delta_deg, receiver_radius)
#     receiverPoints = receiverPoints[receiverPoints.get_sph()[..., 1] < np.pi/2]

#     # execute
#     m2s.input.write_scattering_project(
#         project_path=tmpdir,
#         frequencies=frequencies,
#         sample_path=sample_path,
#         reference_path=reference_path,
#         receiver_coords=receiverPoints,
#         source_coords=source_coords,
#         structural_wavelength=structural_wavelength,
#         model_scale=model_scale,
#         sample_diameter=sample_diameter,
#         symmetry_azimuth=symmetry_azimuth,
#         symmetry_rotational=symmetry_rotational,
#         )

#     # test parameters
#     f = open(os.path.join(tmpdir, 'parameters.json'))
#     paras = json.load(f)
#     source_list = [list(i) for i in list(source_coords.get_cart())]
#     receiver_list = [list(i) for i in list(receiverPoints.get_cart())]
#     parameters = {
#         # project Info
#         "project_title": 'scattering pattern',
#         "mesh2scattering_path": m2s.utils.program_root(),
#         "mesh2scattering_version": m2s.__version__,
#         "bem_version": 'ML-FMM BEM',
#         # Constants
#         "speed_of_sound": float(346.18),
#         "density_of_medium": float(1.1839),
#         # Sample Information, post processing
#         "structural_wavelength": structural_wavelength,
#         "model_scale": model_scale,
#         "sample_diameter": sample_diameter,
#         # symmetry information
#         "symmetry_azimuth": symmetry_azimuth,
#         "symmetry_rotational": symmetry_rotational,
#         # frequencies
#         "num_frequencies": len(frequencies),
#         "min_frequency": frequencies[0],
#         "max_frequency": frequencies[-1],
#         "frequencies": list(frequencies),
#         # Source definition
#         "source_type": 'Point source',
#         "sources_num": len(source_list),
#         "sources": source_list,
#         # Receiver definition
#         "receivers_num": len(receiver_list),
#         "receivers": receiver_list,
#     }
#     npt.assert_array_almost_equal(paras['receivers'], parameters['receivers'])
#     paras['receivers'] = parameters['receivers']
#     npt.assert_equal(paras, parameters)
#     # test folder structure
#     assert os.path.isdir(os.path.join(tmpdir, 'sample'))
#     assert os.path.isdir(os.path.join(tmpdir, 'reference'))
#     assert os.path.isdir(os.path.join(tmpdir, 'sample', 'EvaluationGrids'))
#     assert os.path.isdir(os.path.join(tmpdir, 'reference', 'EvaluationGrids'))
#     assert os.path.isdir(os.path.join(tmpdir, 'sample', 'NumCalc'))
#     assert os.path.isdir(os.path.join(tmpdir, 'reference', 'NumCalc'))
#     assert os.path.isdir(os.path.join(tmpdir, 'sample', 'ObjectMeshes'))
#     assert os.path.isdir(os.path.join(tmpdir, 'reference', 'ObjectMeshes'))
#     assert os.path.isfile(os.path.join(
#         tmpdir, 'sample', 'ObjectMeshes', 'sample.stl'))
#     assert os.path.isfile(os.path.join(
#         tmpdir, 'reference', 'ObjectMeshes', 'reference.stl'))

#     # test sources
#     assert os.path.isdir(
#         os.path.join(tmpdir, 'sample', 'NumCalc', 'source_1'))
#     assert not os.path.isdir(
#         os.path.join(tmpdir, 'sample', 'NumCalc', 'source_2'))
#     assert os.path.isdir(
#         os.path.join(tmpdir, 'reference', 'NumCalc', 'source_1'))
#     assert not os.path.isdir(
#         os.path.join(tmpdir, 'reference', 'NumCalc', 'source_2'))

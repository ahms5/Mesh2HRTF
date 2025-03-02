import pytest
import pyfar as pf
import numpy.testing as npt
from mesh2scattering.input import EvaluationGrid
import numpy as np
import os
import trimesh
import mesh2scattering as m2s
import re
import filecmp


def test_evaluation_grid_initialization():
    coords = pf.Coordinates([0, 1, 2], [3, 4, 5], [6, 7, 8], weights=[1, 1, 1])
    face = np.array([[0, 1, 2]])
    grid = EvaluationGrid(coords, face, "Test Grid")
    npt.assert_almost_equal(grid.coordinates.cartesian, coords.cartesian)
    assert grid.name == "Test Grid"
    npt.assert_almost_equal(grid.weights, [1, 1, 1])


def test_evaluation_grid_invalid_coordinates():
    with pytest.raises(
            ValueError,
            match="coordinates must be a pyfar.Coordinates object."):
        EvaluationGrid(
            "invalid",
            np.array([[0, 1, 2]]),
            "Test Grid")
    with pytest.raises(
            ValueError, match="coordinates must have weights."):
        EvaluationGrid(
            pf.Coordinates([0, 1, 2], [3, 4, 5], [6, 7, 8]),
            np.array([[0, 1, 2]]),
            "Test Grid")


def test_evaluation_grid_invalid_name():
    coords = pf.Coordinates([0, 1, 2], [3, 4, 5], [6, 7, 8], weights=[1, 1, 1])
    with pytest.raises(ValueError, match="name must be a string."):
        EvaluationGrid(coords, np.array([[0, 1, 2]]), 123)


def test_evaluation_grid_invalid_faces():
    coords = pf.Coordinates([0, 1, 2], [3, 4, 5], [6, 7, 8], weights=[1, 1, 1])
    with pytest.raises(ValueError, match="faces must be a np.ndarray."):
        EvaluationGrid(coords, 'string', 'name')
    with pytest.raises(ValueError, match=re.escape(
        "faces must be of shape (n, 3).")):
        EvaluationGrid(coords, np.array([[0]]), 'name')
    with pytest.raises(ValueError, match="faces must be of type int."):
        EvaluationGrid(coords, np.array([[0, 1, 2.5]]), 'name')


def test_from_spherical():
    points = pf.samplings.sph_lebedev(sh_order=10)
    grid = EvaluationGrid.from_spherical(
       points, "Lebedev_N10")
    npt.assert_almost_equal(grid.coordinates.cartesian, points.cartesian)
    assert grid.name == "Lebedev_N10"
    npt.assert_almost_equal(grid.weights, points.weights)
    assert isinstance(grid.faces, np.ndarray)
    npt.assert_array_equal(grid.faces.shape, (336, 3))


@pytest.mark.parametrize("plane", ['xy', 'yz', 'xz'])
def test_from_parallel_to_plane(plane):
    x = np.arange(0, 50, 10)
    y = x
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    if plane == 'xy':
        points = pf.Coordinates(xx, yy, 0, weights=xx)
    elif plane == 'xz':
        points = pf.Coordinates(xx, 0, yy, weights=xx)
    elif plane == 'yz':
        points = pf.Coordinates(0, xx, yy, weights=xx)

    grid = EvaluationGrid.from_parallel_to_plane(
        points, plane, f"{plane}_plane")

    assert grid.name == f"{plane}_plane"
    npt.assert_almost_equal(grid.coordinates.cartesian, points.cartesian)
    npt.assert_almost_equal(grid.weights, points.weights)
    assert isinstance(grid.faces, np.ndarray)
    npt.assert_array_equal(grid.faces.shape, (32, 3))


def test_from_parallel_to_plane_invalid_plane():
    points = pf.Coordinates(0, 0, 0, weights=1)

    with pytest.raises(
            ValueError,
            match="plane must be 'xy', 'yz', or 'xz'."):
        EvaluationGrid.from_parallel_to_plane(
            points, 'xyz', "plane")


@pytest.mark.parametrize("start", [0, 100])
def test_write(start, tmpdir):
    points = pf.samplings.sph_lebedev(sh_order=10)
    grid = EvaluationGrid.from_spherical(
       points, "Lebedev_N10")

    filename = os.path.join(tmpdir, "test_evaluation_grid")

    grid.export_numcalc(filename, start)
    # read and check Nodes
    with open(filename + "/Nodes.txt", "r") as f_id:
        file = f_id.readlines()
    file = "".join(file)

    x = points.x
    y = points.y
    z = points.z
    first_row = f"{points.csize}\n"
    second_row = f"{start} {x[0]} {y[0]} {z[0]}\n"
    assert file.startswith(first_row+second_row)
    assert file.endswith(
        f"\n{start+points.csize-1} {x[-1]} {y[-1]} {z[-1]}\n")

    # read and check Elements
    with open(filename + "/Elements.txt", "r") as f_id:
        file = f_id.readlines()
    assert len(file) == int(file[0]) + 1
    assert int(file[0]) == grid.faces.shape[0]


def test_read_write(tmpdir):
    path = os.path.join(
        m2s.utils.repository_root(), 'tests', 'references', 'Grid')
    mesh_path = os.path.join(path, 'sample.stl')
    mesh = trimesh.load(mesh_path)
    coords = pf.Coordinates()
    coords.cartesian = mesh.vertices
    coords.weights = np.ones(coords.cshape)
    grid = EvaluationGrid(
       coords, mesh.faces, "Lebedev_N10")

    filename = os.path.join(tmpdir, "test_evaluation_grid")

    grid.export_numcalc(filename, start=0)
    assert filecmp.cmp(
        os.path.join(path, 'Elements.txt'),
        os.path.join(tmpdir, 'test_evaluation_grid', 'Elements.txt'),
    )
    assert filecmp.cmp(
        os.path.join(path, 'Nodes.txt'),
        os.path.join(tmpdir, 'test_evaluation_grid', 'Nodes.txt'),
    )

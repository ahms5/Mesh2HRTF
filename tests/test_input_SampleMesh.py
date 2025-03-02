import pytest
from mesh2scattering.input import (
    SurfaceDescription, SurfaceType, SampleMesh, SampleShape,
)
import mesh2scattering as m2s
import trimesh
import os
import filecmp

@pytest.mark.parametrize('surface_type', [
    SurfaceType.PERIODIC, SurfaceType.STOCHASTIC, SurfaceType.FLAT,
    ])
def test_surface_description_initialization(surface_type):
    surface = SurfaceDescription(
        structural_wavelength_x=1.0,
        structural_wavelength_y=2.0,
        structural_depth=3.0,
        surface_type=surface_type,
        model_scale=1.5,
        symmetry_azimuth=[30, 60],
        symmetry_rotational=True,
        comment="Test surface",
    )
    assert surface.structural_wavelength_x == 1.0
    assert surface.structural_wavelength_y == 2.0
    assert surface.surface_type == surface_type
    assert surface.model_scale == 1.5
    assert surface.symmetry_azimuth == [30, 60]
    assert surface.symmetry_rotational is True
    assert surface.comment == "Test surface"
    assert surface.structural_depth == 3.0


def test_surface_description_default_initialization():
    surface = SurfaceDescription()
    assert surface.structural_wavelength_x == 0
    assert surface.structural_wavelength_y == 0
    assert surface.surface_type == SurfaceType.PERIODIC
    assert surface.model_scale == 1
    assert surface.symmetry_azimuth == []
    assert surface.symmetry_rotational is False
    assert surface.comment == ""
    assert surface.structural_depth == 0


def test_surface_description_invalid_structural_wavelength_x():
    with pytest.raises(
            ValueError,
            match="structural_wavelength_x must be a float and >= 0."):
        SurfaceDescription(structural_wavelength_x="invalid")
    with pytest.raises(
            ValueError,
            match="structural_wavelength_x must be a float and >= 0."):
        SurfaceDescription(structural_wavelength_x=-1)


def test_surface_description_invalid_structural_wavelength_y():
    with pytest.raises(
            ValueError,
            match="structural_wavelength_y must be a float and >= 0."):
        SurfaceDescription(structural_wavelength_y="invalid")
    with pytest.raises(
            ValueError,
            match="structural_wavelength_y must be a float and >= 0."):
        SurfaceDescription(structural_wavelength_y=-1)


def test_surface_description_invalid_structural_depth():
    with pytest.raises(
            ValueError,
            match="structural_depth must be a float and >= 0."):
        SurfaceDescription(structural_depth="invalid")
    with pytest.raises(
            ValueError,
            match="structural_depth must be a float and >= 0."):
        SurfaceDescription(structural_depth=-1)


def test_surface_description_invalid_model_scale():
    with pytest.raises(
            ValueError, match="model_scale must be a float and > 0."):
        SurfaceDescription(model_scale="invalid")
    with pytest.raises(
            ValueError, match="model_scale must be a float and > 0."):
        SurfaceDescription(model_scale=-1)


def test_surface_description_invalid_symmetry_azimuth():
    with pytest.raises(ValueError, match="symmetry_azimuth must be a list."):
        SurfaceDescription(symmetry_azimuth="invalid")
    with pytest.raises(
        ValueError,
        match="elements of symmetry_azimuth must be a number between"):
        SurfaceDescription(symmetry_azimuth=[-1])
    with pytest.raises(
        ValueError,
        match="elements of symmetry_azimuth must be a number between"):
        SurfaceDescription(symmetry_azimuth=[361])


def test_surface_description_invalid_symmetry_rotational():
    with pytest.raises(
            ValueError, match="symmetry_rotational must be a bool."):
        SurfaceDescription(symmetry_rotational="invalid")


def test_surface_description_invalid_comment():
    with pytest.raises(ValueError, match="comment must be a string."):
        SurfaceDescription(comment=123)


def test_surface_description_invalid_surface_type():
    with pytest.raises(
            ValueError, match="surface_type must be a SurfaceType."):
        SurfaceDescription(surface_type="invalid")


def test_sample_mesh_initialization(simple_mesh):
    surface_desc = SurfaceDescription(structural_wavelength_x=.08)
    sample_mesh = SampleMesh(
        mesh=simple_mesh,
        surface_description=surface_desc,
        sample_diameter=0.8,
        sample_shape=SampleShape.ROUND,
    )
    assert sample_mesh.mesh == simple_mesh
    assert sample_mesh.surface_description == surface_desc
    assert sample_mesh.sample_diameter == 0.8
    assert sample_mesh.sample_shape == SampleShape.ROUND
    assert sample_mesh.n_repetitions_x == 10
    assert sample_mesh.n_repetitions_y == 0
    assert sample_mesh.n_mesh_elements == 1
    assert sample_mesh.n_mesh_nodes == 3


def test_sample_mesh_invalid_mesh():
    surface_desc = SurfaceDescription()
    with pytest.raises(
            ValueError, match="mesh must be a trimesh.Trimesh object."):
        SampleMesh(
            mesh="invalid",
            surface_description=surface_desc,
            sample_diameter=0.8,
            sample_shape=SampleShape.ROUND,
        )


def test_sample_mesh_invalid_sample_diameter(simple_mesh):
    surface_desc = SurfaceDescription()
    with pytest.raises(
            ValueError,
            match="sample_diameter must be a float or int and >0."):
        SampleMesh(
            mesh=simple_mesh,
            surface_description=surface_desc,
            sample_diameter="invalid",
            sample_shape=SampleShape.ROUND,
        )
    with pytest.raises(
            ValueError,
            match="sample_diameter must be a float or int and >0."):
        SampleMesh(
            mesh=simple_mesh,
            surface_description=surface_desc,
            sample_diameter=-1,
            sample_shape=SampleShape.ROUND,
        )


def test_sample_mesh_invalid_sample_shape(simple_mesh):
    surface_desc = SurfaceDescription()
    with pytest.raises(ValueError, match="sample_shape must be a SampleShape."):
        SampleMesh(
            mesh=simple_mesh,
            surface_description=surface_desc,
            sample_diameter=0.8,
            sample_shape="invalid",
        )


def test_sample_mesh_invalid_surface_description(simple_mesh):
    with pytest.raises(
        ValueError,
        match="surface_description must be a SurfaceDescription object."):
        SampleMesh(
            mesh=simple_mesh,
            surface_description="invalid",
            sample_diameter=0.8,
            sample_shape=SampleShape.ROUND,
        )


def test_sample_mesh_properties(simple_mesh):
    surface_desc = SurfaceDescription()
    sample_mesh = SampleMesh(
        mesh=simple_mesh,
        surface_description=surface_desc,
        sample_diameter=0.8,
        sample_shape=SampleShape.ROUND,
    )
    assert sample_mesh.mesh_faces.shape == (1, 3)
    assert sample_mesh.mesh_vertices.shape == (3, 3)



def test_read_write_mesh(tmpdir):
    path = os.path.join(
        m2s.utils.program_root(), '..', 'tests', 'references', 'Mesh')
    mesh_path = os.path.join(path, 'sample.stl')
    mesh = trimesh.load(mesh_path)
    surface_desc = SurfaceDescription()
    sample_mesh = SampleMesh(
        mesh=mesh,
        surface_description=surface_desc,
        sample_diameter=0.8,
        sample_shape=SampleShape.ROUND,
    )
    sample_mesh.export_numcalc(tmpdir, start=0)
    assert filecmp.cmp(
        os.path.join(path, 'Elements.txt'),
        os.path.join(tmpdir, 'Elements.txt'),
    )
    assert filecmp.cmp(
        os.path.join(path, 'Nodes.txt'),
        os.path.join(tmpdir, 'Nodes.txt'),
    )


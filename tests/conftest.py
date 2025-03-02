import pytest
import trimesh
import numpy as np
import pyfar as pf


@pytest.fixture
def simple_mesh():
    """Return a simple triangle mesh.

    Returns
    -------
    trimesh.Trimesh
        simple triangle
    """
    return trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])

@pytest.fixture
def half_sphere():
    """Return 42th order gaussian sampling for the half sphere and radius 1.

    Returns
    -------
    pf.Coordinates
        half sphere sampling grid
    """
    mics = pf.samplings.sph_gaussian(42)
    # delete lower part of sphere
    return mics[mics.get_sph().T[1] <= np.pi/2]


@pytest.fixture
def quarter_hemisphere_inc():
    """Return 10th order gaussian sampling for the quarter half sphere
    and radius 1.

    Returns
    -------
    pf.Coordinates
        quarter half sphere sampling grid
    """
    incident_directions = pf.samplings.sph_gaussian(10)
    incident_directions = incident_directions[
        incident_directions.get_sph().T[1] <= np.pi/2]
    return incident_directions[
        incident_directions.get_sph().T[0] <= np.pi/2]


@pytest.fixture
def half_hemisphere_inc():
    """Return 10th order gaussian sampling for the quarter half sphere
    and radius 1.

    Returns
    -------
    pf.Coordinates
        quarter half sphere sampling grid
    """
    incident_directions = pf.samplings.sph_gaussian(10)
    incident_directions = incident_directions[
        incident_directions.get_sph().T[1] <= np.pi/2]
    return incident_directions[
        incident_directions.get_sph().T[0] <= np.pi]


@pytest.fixture
def pressure_data_mics(half_sphere):
    """Returns a sound pressure data example, with sound pressure 0 and
    two frequency bins.

    Parameters
    ----------
    half_sphere : pf.Coordinates
        half sphere sampling grid for mics

    Returns
    -------
    pyfar.FrequencyData
        output sound pressure data
    """
    frequencies = [200, 300]
    shape_new = np.append(half_sphere.cshape, len(frequencies))
    return pf.FrequencyData(np.zeros(shape_new), frequencies)


@pytest.fixture
def data_in(
        half_sphere, quarter_hemisphere_inc):
    """Returns a sound pressure data example, with sound pressure 0 and
    two frequency bins.

    Parameters
    ----------
    half_sphere : pf.Coordinates
        half sphere sampling grid for mics
    quarter_hemisphere_inc : pf.Coordinates
        quarter half sphere sampling grid for incident directions

    Returns
    -------
    pyfar.FrequencyData
        output sound pressure data
    """
    frequencies = [200, 300]
    shape_new = np.append(
        quarter_hemisphere_inc.cshape, half_sphere.cshape)
    shape_new = np.append(shape_new, len(frequencies))
    return pf.FrequencyData(np.zeros(shape_new), frequencies)

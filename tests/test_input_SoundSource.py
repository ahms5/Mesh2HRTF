import mesh2scattering as m2s
import pytest
import numpy.testing as npt
import pyfar as pf


def test_sound_source_initialization():
    coords = pf.Coordinates(
        [0, 1, 2], [3, 4, 5], [6, 7, 8], weights=[1, 1, 1])
    source = m2s.input.SoundSource(
        coords, m2s.input.SoundSourceType.POINT_SOURCE)
    assert source.source_type == m2s.input.SoundSourceType.POINT_SOURCE
    npt.assert_almost_equal(
        source.source_coordinates.cartesian, coords.cartesian)
    npt.assert_almost_equal(
        source.source_coordinates.weights, coords.weights)


def test_sound_source_invalid_type():
    coords = pf.Coordinates([0, 1, 2], [3, 4, 5], [6, 7, 8])
    with pytest.raises(
            ValueError, match="source_type must be a SoundSourceType object."):
        m2s.input.SoundSource(coords, "InvalidType")


def test_sound_source_invalid_coordinates():
    with pytest.raises(
            ValueError,
            match="source_coordinates must be a pyfar.Coordinates object."):
        m2s.input.SoundSource(
            "InvalidCoordinates", m2s.input.SoundSourceType.POINT_SOURCE)


def test_sound_source_coordinates_without_weights():
    coords = pf.Coordinates([0, 1, 2], [3, 4, 5], [6, 7, 8])
    coords.weights = None
    with pytest.raises(
            ValueError, match="source_coordinates must contain weights."):
        m2s.input.SoundSource(coords, m2s.input.SoundSourceType.POINT_SOURCE)


def test_sound_source_str():
    coords = pf.Coordinates(
        [0, 1, 2], [3, 4, 5], [6, 7, 8], weights=[1, 1, 1])
    source = m2s.input.SoundSource(
        coords, m2s.input.SoundSourceType.POINT_SOURCE)
    assert str(source) == 'A set of Point sources containing 3 sources.'

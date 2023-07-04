import pytest
import numpy as np
import pyfar as pf
import mesh2scattering as m2s


def test_diffusion_freefield(half_sphere, pressure_data_mics):
    mics = half_sphere
    p_sample = pressure_data_mics.copy()
    p_sample.freq.fill(1)
    d = m2s.utils.diffusion(p_sample, mics.weights)
    np.testing.assert_allclose(d.freq, 1)


@pytest.mark.parametrize("radius",  [
    (1), (10)])
@pytest.mark.parametrize("magnitude",  [
    (1), (10)])
def test_diffusion_freefield_with_theta_0(
        half_sphere, pressure_data_mics, radius, magnitude):
    mics = half_sphere
    spherical = mics.get_sph().T
    mics.set_sph(spherical[0], spherical[1], radius)
    p_sample = pressure_data_mics.copy()
    p_sample.freq.fill(magnitude)
    d = m2s.utils.diffusion(p_sample, mics.weights)
    np.testing.assert_allclose(d.freq, 1)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
@pytest.mark.parametrize("radius",  [
    (1), (10)])
def test_diffusion_freefield_not_one(frequencies, radius):
    # validate with code from itatoolbox
    mics = pf.samplings.sph_equal_angle(10, radius)
    mics.weights = pf.samplings.calculate_sph_voronoi_weights(mics)
    mics = mics[mics.get_sph().T[1] <= np.pi/2]  # delete lower part of sphere
    theta_is_pi = mics.get_sph().T[1] == np.pi/2
    mics.weights[theta_is_pi] /= 2
    data = np.ones(mics.cshape)
    p_sample = pf.FrequencyData(data[..., np.newaxis], [100])
    p_sample.freq[1, :] = 2
    d = m2s.utils.diffusion(p_sample, mics.weights)
    np.testing.assert_allclose(d.freq, 0.9918, atol=0.003)


def test_scattering_freefield_1(
        half_sphere, pressure_data_mics):
    mics = half_sphere
    p_sample = pressure_data_mics.copy()
    p_sample.freq.fill(1)
    p_reference = pressure_data_mics.copy()
    p_sample.freq[5, :] = 0
    p_reference.freq[5, :] = np.sum(p_sample.freq.flatten())/2
    s = m2s.utils.scattering(p_sample, p_reference, mics.weights)
    np.testing.assert_allclose(s.freq, 1)


def test_scattering_freefield_wrong_input(
        half_sphere, pressure_data_mics):
    mics = half_sphere
    p_sample = pressure_data_mics.copy()
    p_reference = pressure_data_mics.copy()

    with pytest.raises(ValueError, match='sample_pressure'):
        m2s.utils.scattering(1, p_reference, mics.weights)
    with pytest.raises(ValueError, match='reference_pressure'):
        m2s.utils.scattering(p_sample, 1, mics.weights)
    with pytest.raises(ValueError, match='microphone_weights'):
        m2s.utils.scattering(p_sample, p_reference, 1)
    with pytest.raises(ValueError, match='cshape'):
        m2s.utils.scattering(
            p_sample[:-2, ...], p_reference, mics.weights)
    with pytest.raises(ValueError, match='microphone_weights'):
        m2s.utils.scattering(
            p_sample, p_reference, mics.weights[:10])
    with pytest.raises(ValueError, match='same frequencies'):
        p_sample.frequencies[0] = 1
        m2s.utils.scattering(p_sample, p_reference, mics.weights)


def test_scattering_freefield_05(
        half_sphere, pressure_data_mics):
    mics = half_sphere
    p_sample = pressure_data_mics.copy()
    p_reference = pressure_data_mics.copy()
    p_sample.freq[7, :] = 1
    p_sample.freq[28, :] = 1
    p_reference.freq[7, :] = 1
    s = m2s.utils.scattering(p_sample, p_reference, mics.weights)
    np.testing.assert_allclose(s.freq, 0.5)


def test_scattering_freefield_0(
        half_sphere, pressure_data_mics):
    mics = half_sphere
    p_sample = pressure_data_mics.copy()
    p_reference = pressure_data_mics.copy()
    p_reference.freq[5, :] = 1
    p_sample.freq[5, :] = 1
    s = m2s.utils.scattering(p_sample, p_reference, mics.weights)
    np.testing.assert_allclose(s.freq, 0)
    assert s.freq.shape[-1] == p_sample.n_bins


def test_scattering_freefield_0_with_incident(
        half_sphere, quarter_half_sphere,
        pressure_data_mics_incident_directions):
    mics = half_sphere
    incident_directions = quarter_half_sphere
    p_sample = pressure_data_mics_incident_directions.copy()
    p_reference = pressure_data_mics_incident_directions.copy()
    p_reference.freq[:, 2, :] = 1
    p_sample.freq[:, 2, :] = 1
    s = m2s.utils.scattering(p_sample, p_reference, mics.weights)
    s_rand = m2s.utils.paris_formula(s, incident_directions)
    np.testing.assert_allclose(s.freq, 0)
    np.testing.assert_allclose(s_rand.freq, 0)
    assert s.freq.shape[-1] == p_sample.n_bins
    assert s.cshape == incident_directions.cshape
    assert s_rand.freq.shape[-1] == p_sample.n_bins


def test_scattering_freefield_1_with_incidence(
        half_sphere, quarter_half_sphere,
        pressure_data_mics_incident_directions):
    mics = half_sphere
    incident_directions = quarter_half_sphere
    p_sample = pressure_data_mics_incident_directions.copy()
    p_reference = pressure_data_mics_incident_directions.copy()
    p_reference.freq[:, 2, :] = 1
    p_sample.freq[:, 3, :] = 1
    s = m2s.utils.scattering(p_sample, p_reference, mics.weights)
    s_rand = m2s.utils.paris_formula(s, incident_directions)
    np.testing.assert_allclose(s.freq, 1)
    np.testing.assert_allclose(s_rand.freq, 1)
    assert s.freq.shape[-1] == p_sample.n_bins
    assert s.cshape == incident_directions.cshape
    assert s_rand.freq.shape[-1] == p_sample.n_bins


def test_freefield_05_with_inci(
        half_sphere, quarter_half_sphere,
        pressure_data_mics_incident_directions):
    mics = half_sphere
    incident_directions = quarter_half_sphere
    p_sample = pressure_data_mics_incident_directions.copy()
    p_reference = pressure_data_mics_incident_directions.copy()
    p_sample.freq[:, 7, :] = 1
    p_sample.freq[:, 28, :] = 1
    p_reference.freq[:, 7, :] = 1
    s = m2s.utils.scattering(p_sample, p_reference, mics.weights)
    s_rand = m2s.utils.paris_formula(s, incident_directions)
    np.testing.assert_allclose(s.freq, 0.5)
    np.testing.assert_allclose(s_rand.freq, 0.5)
    assert s.freq.shape[-1] == p_sample.n_bins
    assert s.cshape == incident_directions.cshape
    assert s_rand.freq.shape[-1] == p_sample.n_bins

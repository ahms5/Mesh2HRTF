import numpy as np
import pyfar as pf


def scattering(
        sample_pressure, reference_pressure, microphone_weights):
    r"""
    Calculate the direction dependent free-field scattering coefficient.

    Uses the Mommertz correlation method [1]_ to calculate the scattering
    coefficient of the input data:

    .. math::
        s = 1 -
            \frac{|\sum_w \underline{p}_{\text{sample}}(\vartheta,\varphi)
            \cdot \underline{p}_{\text{reference}}^*(\vartheta,\varphi)
            \cdot w(\vartheta,\varphi)|^2}
            {\sum_w |\underline{p}_{\text{sample}}(\vartheta,\varphi)|^2
            \cdot w(\vartheta,\varphi) \cdot \sum_w
            |\underline{p}_{\text{reference}}(\vartheta,\varphi)|^2
            \cdot w(\vartheta,\varphi) }

    with the reflected sound pressure of the the sample under investigation
    :math:`\underline{p}_{\text{sample}}`, the reflected sound pressure from
    the reference sample (same dimension as the sample under investigation,
    but with flat surface) :math:`\underline{p}_{\text{reference}}`, the
    area weights of the sampling :math:`w`, and :math:`\vartheta` and
    :math:`\varphi` are the incidence angle and azimuth angles. See
    :py:func:`random` to calculate the random incidence
    scattering coefficient.

    Parameters
    ----------
    sample_pressure : :py:class:`~pyfar.classes.audio.FrequencyData`
        Reflected sound pressure or directivity of the test sample. Its cshape
        needs to be (..., microphone_weights.csize).
    reference_pressure : :py:class:`~pyfar.classes.audio.FrequencyData`
        Reflected sound pressure or directivity of the
        reference sample. Needs to have the same cshape and frequencies as
        `sample_pressure`.
    microphone_weights : numpy.ndarray
        Array containing the area weights for the microphone positions.
        Its shape needs to match the last dimension in the cshape of
        `sample_pressure` and `reference_pressure`.

    Returns
    -------
    scattering_coefficients : :py:class:`~pyfar.classes.audio.FrequencyData`
        The scattering coefficient for each incident direction depending on
        frequency.


    References
    ----------
    .. [1]  E. Mommertz, „Determination of scattering coefficients from the
            reflection directivity of architectural surfaces“, Applied
            Acoustics, Bd. 60, Nr. 2, S. 201-203, June 2000,
            doi: 10.1016/S0003-682X(99)00057-2.

    """
    # check inputs
    if not isinstance(sample_pressure, pf.FrequencyData):
        raise ValueError(
            "sample_pressure has to be a pyfar.FrequencyData object")
    if not isinstance(reference_pressure, pf.FrequencyData):
        raise ValueError(
            "reference_pressure has to be a pyfar.FrequencyData object")
    microphone_weights = np.atleast_1d(
        np.asarray(microphone_weights, dtype=float))
    if sample_pressure.cshape != reference_pressure.cshape:
        raise ValueError(
            "sample_pressure and reference_pressure have to have the "
            "same cshape.")
    if microphone_weights.shape[0] != sample_pressure.cshape[-1]:
        raise ValueError(
            "the last dimension of sample_pressure needs be same as the "
            "microphone_weights.shape.")
    if not np.allclose(
            sample_pressure.frequencies, reference_pressure.frequencies):
        raise ValueError(
            "sample_pressure and reference_pressure have to have the "
            "same frequencies.")

    # calculate according to mommertz correlation method Equation (5)
    p_sample = np.moveaxis(sample_pressure.freq, -1, 0)
    p_reference = np.moveaxis(reference_pressure.freq, -1, 0)
    p_sample_sq = np.abs(p_sample)**2
    p_reference_sq = np.abs(p_reference)**2
    p_cross = p_sample * np.conj(p_reference)

    p_sample_sum = np.sum(microphone_weights * p_sample_sq, axis=-1)
    p_ref_sum = np.sum(microphone_weights * p_reference_sq, axis=-1)
    p_cross_sum = np.sum(microphone_weights * p_cross, axis=-1)

    data_scattering_coefficient \
        = 1 - ((np.abs(p_cross_sum)**2)/(p_sample_sum*p_ref_sum))

    scattering_coefficients = pf.FrequencyData(
        np.moveaxis(data_scattering_coefficient, 0, -1),
        sample_pressure.frequencies)

    return scattering_coefficients


def diffusion(sample_pressure, microphone_weights):
    r"""
    Calculate the free-field diffusion coefficient for each incident direction
    after ISO 17497-2:2012 [1]_. See :py:func:`random_incidence`
    to calculate the random incidence diffusion coefficient.

    .. math::
        d(\vartheta_S,\varphi_S) =
            \frac{(\sum |\underline{p}_{sample}(\vartheta_R,\varphi_R)| \cdot
            N_i)^2 - \sum (|\underline{p}_{sample}(\vartheta_R,\varphi_R)|)^2
            \cdot N_i}
            {(\sum N_i - 1) \cdot \sum
            (|\underline{p}_{sample}(\vartheta_R,\varphi_R)|)^2 \cdot N_i}

    with

    .. math::
        N_i = \frac{A_i}{A_{min}}

    and ``A`` being the area weights ``microphone_weights``.

    Parameters
    ----------
    sample_pressure : pyfar.FrequencyData
        Reflected sound pressure or directivity of the test sample. Its cshape
        need to be (..., #microphones).
    microphone_weights : ndarray
        An array object with all weights for the microphone positions.
        Its cshape need to be (#microphones). Microphone positions need to be
        same for `sample_pressure` and `reference_pressure`.

    Returns
    -------
    diffusion_coefficients : pyfar.FrequencyData
        The diffusion coefficient for each plane wave direction.


    References
    ----------
    .. [1]  ISO 17497-2:2012, Sound-scattering properties of surfaces.
            Part 2: Measurement of the directional diffusion coefficient in a
            free field. Geneva, Switzerland: International Organization for
            Standards, 2012.


    Examples
    --------
    Calculate free-field diffusion coefficients and then the random incidence
    diffusion coefficient.

    >>> import imkar as ik
    >>> diffusion_coefficients = ik.diffusion.coefficient.freefield(
    >>>     sample_pressure, mic_positions.weights)
    >>> random_d = ik.scattering.coefficient.random_incidence(
    >>>     diffusion_coefficients, incident_positions)
    """
    # check inputs
    if not isinstance(sample_pressure, pf.FrequencyData):
        raise ValueError("sample_pressure has to be FrequencyData")
    if not isinstance(microphone_weights, np.ndarray):
        raise ValueError("weights_microphones have to be a numpy.array")
    if not microphone_weights.shape[0] == sample_pressure.cshape[-1]:
        raise ValueError(
            "the last dimension of sample_pressure need be same as the "
            "weights_microphones.shape.")

    # parse angles
    N_i = microphone_weights / np.min(microphone_weights)

    # calculate according to Mommertz correlation method Equation (6)
    p_sample_abs_sq = np.moveaxis(np.abs(sample_pressure.freq)**2, -1, 0)

    p_sample_sum_sq = np.sum(
        p_sample_abs_sq**2 * N_i, axis=-1)
    p_sample_sq_sum = np.sum(
        p_sample_abs_sq * N_i, axis=-1)**2
    n = np.sum(N_i)
    diffusion_array \
        = (p_sample_sq_sum - p_sample_sum_sq) / ((n-1) * p_sample_sum_sq)
    diffusion_coefficients = pf.FrequencyData(
        np.moveaxis(diffusion_array, 0, -1),
        sample_pressure.frequencies)
    diffusion_coefficients.comment = 'diffusion coefficients'

    return diffusion_coefficients


def paris_formula(coefficients, incident_directions):
    r"""
    Calculate the random-incidence coefficient from free-field
    data for several incident directions.

    Uses the Paris formula [2]_.

    .. math::
        c_{rand} = \sum c(\vartheta,\varphi) \cdot cos(\vartheta) \cdot
        w(\vartheta,\varphi)

    with the coefficients :math:`c(\vartheta,\varphi)`, the area
    weights ``w`` from the `incident_directions.weights`,
    and :math:`\vartheta` and :math:`\varphi` are the incidence
    angle and azimuth angles. Note that the incident directions should be
    equally distributed to get a valid result.

    Parameters
    ----------
    coefficients : :py:class:`~pyfar.classes.audio.FrequencyData`
        Scattering coefficients for different incident directions. Its cshape
        needs to be (..., `incident_directions.csize`)
    incident_directions : :py:class:`~pyfar.classes.coordinates.Coordinates`
        Defines the incidence directions of each `scattering_coefficients`
        in a :py:class:`~pyfar.classes.coordinates.Coordinates` object.
        Its cshape needs to match
        the last dimension of `scattering_coefficients`.
        Points contained in `incident_directions` must have the same radii.
        The weights need to reflect the area ``incident_directions.weights``.

    Returns
    -------
    random_coefficient : pyfar.FrequencyData
        The random-incidence coefficient depending on frequency.

    References
    ----------
    .. [2]  H. Kuttruff, Room acoustics, Sixth edition. Boca Raton:
            CRC Press/Taylor & Francis Group, 2017.
    """
    if not isinstance(coefficients, pf.FrequencyData):
        raise ValueError("coefficients has to be FrequencyData")
    if not isinstance(incident_directions, pf.Coordinates):
        raise ValueError("incident_directions have to be None or Coordinates")
    if incident_directions.cshape[0] != coefficients.cshape[-1]:
        raise ValueError(
            "the last dimension of coefficients needs be same as "
            "the incident_directions.cshape.")

    theta = incident_directions.get_sph().T[1]
    weight = np.cos(theta) * incident_directions.weights
    norm = np.sum(weight)
    coefficients_freq = np.swapaxes(coefficients.freq, -1, -2)
    random_coefficient = pf.FrequencyData(
        np.sum(coefficients_freq*weight/norm, axis=-1),
        coefficients.frequencies,
        comment='random-incidence coefficient'
    )
    return random_coefficient

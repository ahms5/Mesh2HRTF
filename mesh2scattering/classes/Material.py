import numpy as np
import os
import pyfar as pf


class BoundaryCondition:
    """Defines a boundary condition for a mesh, e.g material property.

    NOTE: Frequency dependent boundary conditions can only be specified for
    ADMI and IMPE but not for PRES and VELO

    Parameters
    ----------
    values : pf.FrequencyData
        Defines the boundary condition values with its frequencies.
    kind : str
        the kind of boundary condition.
    comment : str
        A comment that is written to the beginning of the material file.
    """

    _values: pf.FrequencyData = None
    _kind: str = None
    _comment: str = None

    def __init__(self, values: pf.FrequencyData, kind: str, comment: str=None):
        """Initializes the Material object.

        Parameters
        ----------
        values : pf.FrequencyData
            Defines the boundary condition values with its frequencies.
        kind : str
            the kind of boundary condition.
        comment : str
            A comment that is written to the beginning of the material file.
        """
        self.values = values
        self.kind = kind
        self.comment = comment

    @property
    def values(self):
        """Defines the boundary condition values with its frequencies.

        Returns
        -------
        pf.FrequencyData
            The boundary condition values with its frequencies.
        """
        return self._values

    @values.setter
    def values(self, values):
        if not isinstance(values, pf.FrequencyData):
            raise ValueError("values must be a pyfar.FrequencyData object.")
        self._values = values

    @property
    def kind(self):
        """Defines the kind of boundary condition.

        - ``'PRES'``
            A pressure boundary condition can be used to force a certain
            pressure on the boundary of the mesh. E.g., a pressure of 0 would
            define a sound soft boundary.
            Cannot be frequency dependent, this means frequency must be 0.
        - ``'VELO'``
            A velocity boundary condition can be used to force a certain
            velocity on the boundary of the mesh. E.g., a velocity of 0 would
            define a sound hard boundary.
            Cannot be frequency dependent, this means frequency must be 0.
        - ``'ADMI'``
            A normalized admittance boundary condition can be used to define
            arbitrary boundaries. The admittance must be normalized, i.e.,
            admittance/(rho*c) has to be provided, which rho being the density
            of air in kg/m**3 and c the speed of sound in m/s.
        - ``'IMPE'``
            A normalized impedance boundary condition can be used to define
            arbitrary boundaries. The impedance must be normalized, i.e.,
            impedance/(rho*c) has to be provided, which rho being the density
            of air in kg/m**3 and c the speed of sound in m/s.

        Returns
        -------
        str
            kind of boundary condition
        """
        return self._kind

    @kind.setter
    def kind(self, kind):
        if not isinstance(kind, str):
            raise ValueError("kind must be a string.")
        if kind not in ["ADMI", "PRES", "VELO", "IMPE"]:
            raise ValueError(
                "kind must be 'ADMI', 'IMPE', 'PRES', or "
                f"'VELO' but is {kind}.")
        self._kind = kind

    @property
    def comment(self):
        """A comment that is written to the beginning of the material file.

        Returns
        -------
        str
            A comment for the material.
        """
        return self._comment

    @comment.setter
    def comment(self, comment):
        if not isinstance(comment, str):
            raise ValueError("comment must be a string.")
        self._comment = comment

    def write(self, path):
        """Writes the material to a file.

        Parameters
        ----------
        path : str
            The path to the file where the material should be written to.
        """
        if not isinstance(path, str):
            raise ValueError("path must be a string.")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))



def read_material_data(materials):
    """
    Read material data from file.

    Mesh2HRTF supports non-rigid boundary conditions in the form of text files.

    Parameters
    ----------
    materials : dict
        Dictionary containing the materials. The keys are the names of the
        materials and the values are dictionaries containing the path to the
        material file.

    Returns
    -------
    materials : dict
        Dictionary containing the materials. The keys are the names of the
        materials and the values are dictionaries containing the path to the
        material file and the boundary condition, frequencies, real, and
        imaginary values.
    """

    for material in materials:
        # current material file
        file = materials[material]["path"]
        # check if the file exists
        if file is None:
            continue

        # initialize data
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
        # check if frequency vector is value
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


def write_material(
        filename: str, kind: str, data: pf.FrequencyData,
        comment: str=None)->None:
    """
    Write boundary condition to file for NumCalc.

    NumCalc supports non-rigid boundary conditions in the form of text files.

    Parameters
    ----------
    filename : str
        Name of the material file that is written to disk. Must end with ".csv"
    kind : str
        Defines the kind of boundary condition

        ``"pressure"``
            A pressure boundary condition can be used to force a certain
            pressure on the boundary of the mesh. E.g., a pressure of 0 would
            define a sound soft boundary.
        ``"velocity"``
            A velocity boundary condition can be used to force a certain
            velocity on the boundary of the mesh. E.g., a velocity of 0 would
            define a sound hard boundary.
        ``admittance``
            A normalized admittance boundary condition can be used to define
            arbitrary boundaries. The admittance must be normalized, i.e.,
            admittance/(rho*c) has to be provided, which rho being the density
            of air in kg/m**3 and c the speed of sound in m/s.
    data : pyfar.FrequencyData
        The boundary condition values with its frequencies.
    comment : str, optional
        A comment that is written to the beginning of the material file. The
        default ``None`` does omit the comment.

    Notes
    -----
    NumCalc performs an interpolation in case the boundary condition is
    required at frequencies that are not specified. The interpolation is linear
    between the lowest and highest provided frequency and uses the nearest
    neighbor outside this range.
    """

    # check input
    if not isinstance(data, pf.FrequencyData):
        raise ValueError("data must be a pyfar.FrequencyData object.")
    if not isinstance(filename, (str, Path)):
        raise ValueError("filename must be a string or Path.")
    if not str(filename).endswith(".csv"):
        raise ValueError("The filename must end with .csv.")
    if not isinstance(comment, (str, type(None))):
        raise ValueError("comment must be a string or None.")
    if not isinstance(kind, str):
        raise ValueError("kind must be a string.")
    if kind not in ["admittance", "pressure", "velocity"]:
        raise ValueError("kind must be admittance, pressure, or velocity.")

    # write the comment
    file = ""
    if comment is not None:
        file += "# " + comment + "\n#\n"

    # write the kind of boundary condition
    file += ("# Keyword to define the boundary condition:\n"
             "# ADMI: Normalized admittance boundary condition\n"
             "# PRES: Pressure boundary condition\n"
             "# VELO: Velocity boundary condition\n"
             "# NOTE: Mesh2HRTF expects normalized admittances, i.e., "
             "admittance/(rho*c).\n"
             "#       rho is the density of air and c the speed of sound. "
             "The normalization is\n"
             "#       beneficial because a single material file can be used "
             "for simulations\n"
             "#       with differing speed of sound and density of air.\n")

    if kind == "admittance":
        file += "ADMI\n"
    elif kind == "pressure":
        file += "PRES\n"
    elif kind == "velocity":
        file += "VELO\n"
    else:
        raise ValueError("kind must be admittance, pressure, or velocity")

    file += ("#\n"
             "# Frequency curve:\n"
             "# NumCalc performs an interpolation in case the boundary "
             "condition is required\n"
             "# at frequencies that are not specified. The interpolation is "
             "linear between the\n"
             "# lowest and highest provided frequency and uses the nearest "
             "neighbor outside\n"
             "# this range.\n"
             "#\n"
             "# Frequency in Hz, real value, imaginary value\n")

    # write data
    for i in range(data.n_bins):
        d = data.freq[0, i]
        file += f"{data.frequencies[i]}, {np.real(d)}, {np.imag(d)}\n"

    # write to file
    with open(filename, "w") as f_id:
        f_id.write(file)

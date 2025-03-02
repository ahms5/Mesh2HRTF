"""
This module provides functions to write input files for NumCalc.
"""
import pyfar as pf
from enum import Enum

class SoundSourceType(Enum):
    """Defines the type of a sound source.

    Can be a point source or a plane wave.
    """

    POINT_SOURCE = "Point source"
    """Point source sound source type."""

    PLANE_WAVE = "Plane wave"
    """Plane wave sound source type."""

class SoundSource():
    """Defines a sound source including its type and coordinates.

    Returns
    -------
    SoundSource
        sound source object.
    """

    _source_type: SoundSourceType = None
    _source_coordinates: pf.Coordinates = None

    def __init__(
            self,
            source_coordinates: pf.Coordinates,
            source_type: SoundSourceType=SoundSourceType.POINT_SOURCE,
            ) -> None:
        """Initializes the SoundSource object.

        Parameters
        ----------
        source_coordinates : pf.Coordinates
            source coordinates, if the sound source type is Point source,
            it reflects the positions in space, and it the source type is
            plane wave it reflects the direction of the sound wave.
        source_type : SoundSourceType, optional
            sound source type, see SoundSourceType, by default
            SoundSourceType.POINT_SOURCE

        Returns
        -------
        SoundSource
            sound source object.
        """
        if not isinstance(source_type, SoundSourceType):
            raise ValueError(
                "source_type must be a SoundSourceType object.")
        self._source_type = source_type

        self.source_coordinates = source_coordinates

    @property
    def source_type(self):
        """Defines the source type.

        Can be a point source or a plane wave.

        Returns
        -------
        SoundSourceType
            The source type.
        """
        return self._source_type

    @property
    def source_coordinates(self):
        """Defines the source coordinates in meter.

        Returns
        -------
        pf.Coordinates
            The source coordinates in meter.
        """
        return self._source_coordinates

    @source_coordinates.setter
    def source_coordinates(self, source_coordinates):
        if not isinstance(source_coordinates, pf.Coordinates):
            raise ValueError(
                "source_coordinates must be a pyfar.Coordinates object.")
        if source_coordinates.weights is None:
            raise ValueError("source_coordinates must contain weights.")
        self._source_coordinates = source_coordinates

    def __str__(self):
        """Return a string representation of the SoundSource object.
        """
        return (
            f'A set of {self.source_type.value}s containing '
            f'{self.source_coordinates.csize} sources.')

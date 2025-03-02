import trimesh
from enum import Enum

class SurfaceType(Enum):
    """Defines the type of a sample mesh.

    Can be a trimesh object or a path to a stl file.
    """

    PERIODIC = "Periodic"
    STOCHASTIC = "Stochastic"

class SampleShape(Enum):
    """Defines the shape of a sample mesh.

    Can be round or square.
    """

    ROUND = "Round"
    SQUARE = "Square"


class SurfaceDescription():
    """Initializes the SurfaceDescription object.

    Parameters
    ----------
    structural_wavelength_x : float, optional
        structural wavelength in x direction, by default 0.
    structural_wavelength_y : float, optional
        structural wavelength in y direction, by default 0.
    surface_type : SurfaceType, optional
        surface type, by default SurfaceType.PERIODIC.
    model_scale : float, optional
        model scale, by default 1.
    symmetry_azimuth : list, optional
        azimuth symmetry, by default [].
    symmetry_rotational : bool, optional
        rotational symmetry, by default False.
    comment : str, optional
        comment, by default "".
    """

    _structural_wavelength_x: float = 0
    _structural_wavelength_y: float = 0
    _model_scale: float = 1
    _symmetry_azimuth: list = []
    _symmetry_rotational: bool = False
    _surface_type: SurfaceType = SurfaceType.PERIODIC
    _comment: str = ""

    def __init__(
            self,
            structural_wavelength_x: float=0,
            structural_wavelength_y: float=0,
            surface_type: SurfaceType=SurfaceType.PERIODIC,
            model_scale: float=1,
            symmetry_azimuth: list=[],
            symmetry_rotational: bool=False,
            comment: str="") -> None:
        """Initializes the SurfaceDescription object.

        Parameters
        ----------
        structural_wavelength_x : float, optional
            structural wavelength in x direction, by default 0.
        structural_wavelength_y : float, optional
            structural wavelength in y direction, by default 0.
        surface_type : SurfaceType, optional
            surface type, by default SurfaceType.PERIODIC.
        model_scale : float, optional
            model scale, by default 1.
        symmetry_azimuth : list, optional
            azimuth symmetry, by default [].
        symmetry_rotational : bool, optional
            rotational symmetry, by default False.
        comment : str, optional
            comment, by default "".

        Returns
        -------
        SurfaceDescription
            surface description object.
        """
        if not isinstance(structural_wavelength_x, (int, float)):
            raise ValueError("structural_wavelength_x must be a float.")
        if not isinstance(structural_wavelength_y, (int, float)):
            raise ValueError("structural_wavelength_y must be a float.")
        if not isinstance(model_scale, (int, float)):
            raise ValueError("model_scale must be a float.")
        if not isinstance(symmetry_azimuth, list):
            raise ValueError("symmetry_azimuth must be a list.")
        if not isinstance(symmetry_rotational, bool):
            raise ValueError("symmetry_rotational must be a bool.")
        if not isinstance(comment, str):
            raise ValueError("comment must be a string.")
        if not isinstance(surface_type, SurfaceType):
            raise ValueError("surface_type must be a SurfaceType.")

        self._structural_wavelength_x = structural_wavelength_x
        self._structural_wavelength_y = structural_wavelength_y
        self._model_scale = model_scale
        self._symmetry_azimuth = symmetry_azimuth
        self._symmetry_rotational = symmetry_rotational
        self._comment = comment
        self._surface_type = surface_type


    @property
    def structural_wavelength_x(self):
        """Defines the structural wavelength in x direction.

        Returns
        -------
        float
            The structural wavelength in x direction.
        """
        return self._structural_wavelength_x

    @property
    def structural_wavelength_y(self):
        """Defines the structural wavelength in y direction.

        Returns
        -------
        float
            The structural wavelength in y direction.
        """
        return self._structural_wavelength_y

    @property
    def surface_type(self):
        """Defines the surface type.

        Returns
        -------
        SurfaceType
            The surface type.
        """
        return self._surface_type

    @property
    def model_scale(self):
        """Defines the model scale.

        Returns
        -------
        float
            The model scale.
        """
        return self._model_scale

    @property
    def symmetry_azimuth(self):
        """Defines the azimuth symmetry.

        Returns
        -------
        list
            The azimuth symmetry.
        """
        return self._symmetry_azimuth

    @property
    def symmetry_rotational(self):
        """Defines the rotational symmetry.

        Returns
        -------
        bool
            The rotational symmetry.
        """
        return self._symmetry_rotational

    @property
    def comment(self):
        """Defines the comment.

        Returns
        -------
        str
            The comment.
        """
        return self._comment


class SampleMesh():
    _mesh:trimesh.Trimesh = None
    _surface_description: SurfaceDescription = None
    _sample_diameter: float = 0.8
    _sample_shape: SampleShape = SampleShape.ROUND

    def __init__(
            self, mesh: trimesh.Trimesh,
            surface_description: SurfaceDescription,
            sample_diameter: float=0.8,
            sample_shape: SampleShape=SampleShape.ROUND) -> None:
        """Initializes the SampleMesh object.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            trimesh object representing the sample mesh.
        surface_description : SurfaceDescription
            surface description of the sample mesh.
        sample_diameter : float, optional
            diameter of the sample, by default 0.8
        sample_shape : str, optional
            shape of the sample, by default 'round'

        Returns
        -------
        SampleMesh
            sample mesh object.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("mesh must be a trimesh.Trimesh object.")
        if not isinstance(sample_diameter, (int, float)):
            raise ValueError("sample_diameter must be a float or int.")
        if not isinstance(sample_shape, SampleShape):
            raise ValueError("sample_shape must be a SampleShape.")
        if not isinstance(surface_description, SurfaceDescription):
            raise ValueError(
                "surface_description must be a SurfaceDescription object.")

        self._mesh = mesh
        self._surface_description = surface_description
        self._sample_diameter = sample_diameter
        self._sample_shape = sample_shape

    @property
    def mesh(self):
        """Defines the sample mesh.

        Returns
        -------
        trimesh.Trimesh
            The sample mesh.
        """
        return self._mesh


    @property
    def sample_diameter(self):
        """Defines the diameter of the sample.

        Returns
        -------
        float
            The diameter of the sample.
        """
        return self._sample_diameter


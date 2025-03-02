import trimesh
import os
from enum import Enum

class SurfaceType(Enum):
    """Defines the type of a sample mesh.

    Can be a trimesh object or a path to a stl file.
    """

    PERIODIC = "Periodic"
    """The surface is periodic."""

    STOCHASTIC = "Stochastic"
    """The surface is stochastic."""

    FLAT = "Flat"
    """The surface is flat, basically for the reference surface."""

class SampleShape(Enum):
    """Defines the shape of a sample mesh.

    Can be round or square.
    """

    ROUND = "Round"
    """The sample shape is round."""

    SQUARE = "Square"
    """The sample shape is square."""


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
    _structural_depth: float = 0

    def __init__(
            self,
            structural_wavelength_x: float=0,
            structural_wavelength_y: float=0,
            structural_depth: float=0,
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
        structural_depth : float, optional
            structural depth in meters, by default 0.
        surface_type : SurfaceType, optional
            surface type, by default SurfaceType.PERIODIC.
        model_scale : float, optional
            model scale, by default 1.
        symmetry_azimuth : list, optional
            along which azimuth angles in degree is the surface symmetric,
            by default [].
        symmetry_rotational : bool, optional
            rotational symmetry, by default False.
        comment : str, optional
            comment, by default "".

        Returns
        -------
        SurfaceDescription
            surface description object.
        """
        if not isinstance(structural_wavelength_x, (int, float)) or \
             structural_wavelength_x < 0:
            raise ValueError(
                "structural_wavelength_x must be a float and >= 0.")
        if not isinstance(structural_wavelength_y, (int, float)) or \
             structural_wavelength_y < 0:
            raise ValueError(
                "structural_wavelength_y must be a float and >= 0.")
        if not isinstance(model_scale, (int, float)) or model_scale <= 0:
            raise ValueError("model_scale must be a float and > 0.")
        if not isinstance(symmetry_azimuth, list):
            raise ValueError("symmetry_azimuth must be a list.")
        for angle in symmetry_azimuth:
            if not isinstance(angle, (int, float)) or angle < 0 or angle > 360:
                raise ValueError(
                    "elements of symmetry_azimuth must be a number between "
                    "0° and 360°.")
        if not isinstance(symmetry_rotational, bool):
            raise ValueError("symmetry_rotational must be a bool.")
        if not isinstance(comment, str):
            raise ValueError("comment must be a string.")
        if not isinstance(surface_type, SurfaceType):
            raise ValueError("surface_type must be a SurfaceType.")
        if not isinstance(structural_depth, (int, float)) or \
                structural_depth < 0:
            raise ValueError("structural_depth must be a float and >= 0.")

        self._structural_wavelength_x = structural_wavelength_x
        self._structural_wavelength_y = structural_wavelength_y
        self._model_scale = model_scale
        self._symmetry_azimuth = symmetry_azimuth
        self._symmetry_rotational = symmetry_rotational
        self._comment = comment
        self._surface_type = surface_type
        self._structural_depth = structural_depth


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
    def structural_depth(self):
        """Defines the structural depth.

        Returns
        -------
        float
            The structural depth.
        """
        return self._structural_depth

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
    """

    _mesh:trimesh.Trimesh = None
    _surface_description: SurfaceDescription = None
    _sample_diameter: float = 0.8
    _sample_shape: SampleShape = SampleShape.ROUND
    _n_repetitions_x: int = 0
    _n_repetitions_y: int = 0

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
        if not isinstance(
                sample_diameter, (int, float)) or sample_diameter <= 0:
            raise ValueError("sample_diameter must be a float or int and >0.")
        if not isinstance(sample_shape, SampleShape):
            raise ValueError("sample_shape must be a SampleShape.")
        if not isinstance(surface_description, SurfaceDescription):
            raise ValueError(
                "surface_description must be a SurfaceDescription object.")

        self._mesh = mesh
        self._surface_description = surface_description
        self._sample_diameter = sample_diameter
        self._sample_shape = sample_shape
        # calculate Number of repetitions in x and y direction
        Lambda_x = surface_description.structural_wavelength_x
        Lambda_y = surface_description.structural_wavelength_y
        self._n_repetitions_x = (
            sample_diameter / Lambda_x) if Lambda_x > 0 else 0
        self._n_repetitions_y = (
            sample_diameter / Lambda_y) if Lambda_y > 0 else 0

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
    def surface_description(self):
        """Defines the surface description.

        Returns
        -------
        SurfaceDescription
            The surface description.
        """
        return self._surface_description

    @property
    def sample_diameter(self):
        """Defines the diameter of the sample.

        Returns
        -------
        float
            The diameter of the sample.
        """
        return self._sample_diameter


    @property
    def sample_shape(self):
        """Defines the shape of the sample.

        Returns
        -------
        SampleShape
            The shape of the sample.
        """
        return self._sample_shape

    @property
    def mesh_faces(self):
        """Defines the faces of the mesh.

        Returns
        -------
        numpy.ndarray
            The faces of the mesh.
        """
        return self._mesh.faces

    @property
    def mesh_vertices(self):
        """Defines the vertices of the mesh.

        Returns
        -------
        numpy.ndarray
            The vertices of the mesh.
        """
        return self._mesh.vertices

    @property
    def n_mesh_elements(self):
        """Number of mesh elements/faces.

        Returns
        -------
        int
            number of mesh elements/faces.
        """
        return self.mesh_faces.shape[0]

    @property
    def n_mesh_nodes(self):
        """Number of mesh nodes/vertices.

        Returns
        -------
        int
            number of mesh nodes/vertices.
        """
        return self.mesh_vertices.shape[0]

    @property
    def n_repetitions_x(self):
        """Defines the number of repetitions in x direction.

        Returns
        -------
        int
            The number of repetitions in x direction.
        """
        return self._n_repetitions_x

    @property
    def n_repetitions_y(self):
        """Defines the number of repetitions in y direction.

        Returns
        -------
        int
            The number of repetitions in y direction.
        """
        return self._n_repetitions_y

    def export_numcalc(self, folder_path, start=200000):
        """
        Write mesh to Mesh2HRTF input format.

        Mesh2HRTF meshes consist of two text files Nodes.txt and Elements.txt.
        The Nodes.txt file contains the coordinates of the vertices and the
        Elements.txt file contains the indices of the vertices that form the
        faces of the mesh.

        Parameters
        ----------
        folder_path : str
            Path to the directory where the mesh is saved. The mesh is saved in
            'Nodes.txt' and 'Elements.txt' files.
        start : int, optional
            The nodes and elements of the mesh are numbered and the first
            element will have the number `start`. In Mesh2HRTF, each Node
            must have a unique number. The nodes/elements of the mesh for
            which the HRTFs are simulated start at 1. Thus `start` must
            at least be greater than the number of nodes/elements in the mesh.

        """
        vertices = self.mesh_vertices
        faces = self.mesh_faces

        # check output directory
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        # write nodes
        N = int(self.mesh.vertices.shape[0])
        start = int(start)

        nodes = f"{N}\n"
        for nn in range(N):
            nodes += (f"{int(start + nn)} "
                    f"{vertices[nn, 0]} "
                    f"{vertices[nn, 1]} "
                    f"{vertices[nn, 2]}\n")

        with open(os.path.join(folder_path, "Nodes.txt"), "w") as f_id:
            f_id.write(nodes)

        # write elements
        N = int(faces.shape[0])
        elements = f"{N}\n"
        for nn in range(N):
            elements += (
                f"{int(start + nn)} "
                f"{faces[nn, 0] + start} "
                f"{faces[nn, 1] + start} "
                f"{faces[nn, 2] + start} "
                "0 0 0\n")

        with open(os.path.join(folder_path, "Elements.txt"), "w") as f_id:
            f_id.write(elements)

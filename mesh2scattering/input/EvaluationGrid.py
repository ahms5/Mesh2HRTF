import pyfar as pf
import numpy as np
import os
from scipy.spatial import Delaunay, ConvexHull


class EvaluationGrid():
    """Initialize the EvaluationGrid object.

    Note that a EvaluationGrid object is always triangulated.
    Please use the corresponding constructor to use the correct triangulation.

    Parameters
    ----------
    coordinates : pf.Coordinates
        The coordinates of the evaluation grid.
    faces : np.ndarray
        The faces of the evaluation grid.
    name : str
        The name of the evaluation grid.

    Examples
    --------
    Generate a spherical sampling grid with pyfar and write it to the current
    working directory

    .. plot::

        >>> import mesh2scattering as m2s
        >>> import pyfar as pf
        >>>
        >>> points = pf.samplings.sph_lebedev(sh_order=10)
        >>> grid = m2s.input.EvaluationGrid.from_spherical(
        ...     points, "Lebedev_N10")
        >>> grid.write(".")
    """

    _coordinates: pf.Coordinates
    _name: str

    def __init__(
            self, coordinates: pf.Coordinates, faces:np.ndarray, name: str):
        """Initialize the EvaluationGrid object.

        Parameters
        ----------
        coordinates : pf.Coordinates
            The coordinates of the evaluation grid.
        faces : np.ndarray of ints
            The faces of the evaluation grid. must be of shape (n, 3).
        name : str
            The name of the evaluation grid.
        """
        if not isinstance(coordinates, pf.Coordinates):
            raise ValueError("coordinates must be a pyfar.Coordinates object.")
        if coordinates.weights is None:
            raise ValueError("coordinates must have weights.")
        if not isinstance(name, str):
            raise ValueError("name must be a string.")
        if not isinstance(faces, np.ndarray):
            raise ValueError("faces must be a np.ndarray.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must be of shape (n, 3).")
        faces = np.atleast_2d(faces)
        if not isinstance(faces[0, 0], (np.int32, int, np.int64)):
            raise ValueError("faces must be of type int.")

        self._coordinates = coordinates
        self._faces = faces
        self._name = name

    @classmethod
    def from_spherical(cls, coordinates, name):
        """Return the evaluation grid with the spherical coordinates.

        Parameters
        ----------
        coordinates : pf.Coordinates
            The coordinates of the evaluation grid.
        name : str
            The name of the evaluation grid.

        Returns
        -------
        EvaluationGrid
            The evaluation grid with the spherical coordinates.
        """
        points = coordinates.cartesian
        tri = ConvexHull(points)
        faces = tri.simplices
        return cls(coordinates, faces, name)

    @classmethod
    def from_parallel_to_plane(cls, coordinates, plane, name):
        """Evaluation grid build from a sampling parallel to
        'xy', 'yz' or 'xz' plane.

        Parameters
        ----------
        coordinates : pf.Coordinates
            The coordinates of the evaluation grid.
        plane : "xy", "yz", "xz"
            In case all values of the evaluation grid are constant for one
            dimension, this dimension has to be discarded during the
            triangulation. E.g. if all points have a z-value of 0 (or
            any other constant), plane must be "xy".
        name : str
            The name of the evaluation grid.

        Returns
        -------
        EvaluationGrid
            The evaluation grid with the parallel to the xy plane.
        """
        if plane == "xy":
            mask = (0, 1)
        elif plane == "yz":
            mask = (1, 2)
        elif plane == "xz":
            mask = (0, 2)
        else:
            raise ValueError("plane must be 'xy', 'yz', or 'xz'.")

        points = coordinates.cartesian[:, mask]
        tri = Delaunay(points)
        return cls(coordinates, tri.simplices, name)

    @property
    def coordinates(self) -> pf.Coordinates:
        """Return the coordinates of the evaluation grid.

        Returns
        -------
        pf.Coordinates
            The coordinates of the evaluation grid.
        """
        return self._coordinates

    @property
    def name(self) -> str:
        """Return the name of the evaluation grid.

        Returns
        -------
        str
            The name of the evaluation grid.
        """
        return self._name

    @property
    def weights(self):
        """Return the weights of the evaluation grid.

        Returns
        -------
        np.ndarray
            The weights of the evaluation grid.
        """
        return self._coordinates.weights

    @property
    def faces(self):
        """Return the faces of the evaluation grid.

        Returns
        -------
        np.ndarray
            The faces of the evaluation grid.
        """
        return self._faces


    def export_numcalc(
            self, folder_path:str, start:int=200000):
        """
        Write evaluation grid for use in Mesh2HRTF.

        Mesh2HRTF evaluation grids consist of the two text files Nodes.txt and
        Elements.txt. Evaluations grids are always triangulated.

        Parameters
        ----------
        folder_path : str, Path
            folder path under which the evaluation grid is saved. If the
            folder does not exist, it is created.
        start : int, optional
            The nodes and elements of the evaluation grid are numbered and the
            first element will have the number `start`. In Mesh2HRTF, each Node
            must have a unique number. The nodes/elements of the mesh for
            which the HRTFs are simulated start at 1. Thus `start` must at
            least be greater than the number of
            nodes/elements in the evaluation grid.

        """
        points = self._coordinates.cartesian
        faces = self._faces

        # check output directory
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        # write nodes
        N = int(points.shape[0])
        start = int(start)

        nodes = f"{N}\n"
        for nn in range(N):
            nodes += (f"{int(start + nn)} "
                    f"{points[nn, 0]} "
                    f"{points[nn, 1]} "
                    f"{points[nn, 2]}\n")

        with open(os.path.join(folder_path, "Nodes.txt"), "w") as f_id:
            f_id.write(nodes)

        # write elements
        N = int(faces.shape[0])
        elems = f"{N}\n"
        for nn in range(N):
            elems += (f"{int(start + nn)} "
                    f"{faces[nn, 0] + start} "
                    f"{faces[nn, 1] + start} "
                    f"{faces[nn, 2] + start} "
                    "2 0 1\n")

        with open(os.path.join(folder_path, "Elements.txt"), "w") as f_id:
            f_id.write(elems)

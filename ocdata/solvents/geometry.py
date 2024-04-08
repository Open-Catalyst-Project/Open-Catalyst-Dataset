import numpy as np
from ase import Atom
from noise import snoise2, pnoise2
from noise_randomized import snoise2 as snoise2r, randomize
import warnings


class Geometry:
    """Base class for geometries.

    :param periodic_boundary_condition: self-explanatory
    :type periodic_boundary_condition: array_like
    :param minimum_image_convention: use the minimum image convention for
                                     bookkeeping how the particles interact
    :type minimum_image_convention: bool
    """

    def __init__(self, periodic_boundary_condition=(False, False, False),
                 minimum_image_convention=True):
        self.minimum_image_convention = minimum_image_convention
        self.periodic_boundary_condition = periodic_boundary_condition
        pass

    def __call__(self, atoms):
        """The empty geometry. False because we define no particle to be
        in the dummy geometry.

        :param atoms: atoms object from ase.Atom that is being modified
        :type atoms: ase.Atom obj
        :returns: ndarray of bools telling which atoms to remove
        :rtype: ndarray of bool
        """
        return np.zeros(len(atoms), dtype=np.bool)

    @staticmethod
    def distance_point_line(vec, point_line, point_ext):
        """Returns the (shortest) distance between a line parallel to
        a normal vector 'vec' through point 'point_line' and an external
        point 'point_ext'.

        :param vec: unit vector parallel to line
        :type vec: ndarray
        :param point_line: point on line
        :type point_line: ndarray
        :param point_ext: external points
        :type point_ext: ndarray
        :return: distance between line and external point(s)
        :rtype: ndarray
        """
        return np.linalg.norm(np.cross(vec, point_ext - point_line), axis=1)

    @staticmethod
    def distance_point_plane(vec, point_plane, point_ext):
        """Returns the (shortest) distance between a plane with normal vector
        'vec' through point 'point_plane' and a point 'point_ext'.

        :param vec: normal vector of plane
        :type vec: ndarray
        :param point_plane: point on line
        :type point_plane: ndarray
        :param point_ext: external point(s)
        :type point_ext: ndarray
        :return: distance between plane and external point(s)
        :rtype: ndarray
        """
        vec = np.atleast_2d(vec)    # Ensure n is 2d
        return np.abs(np.einsum('ik,jk->ij', point_ext - point_plane, vec))

    @staticmethod
    def vec_and_point_to_plane(vec, point):
        """Returns the (unique) plane, given a normal vector 'vec' and a
        point 'point' in the plane. ax + by + cz - d = 0

        :param vec: normal vector of plane
        :type vec: ndarray
        :param point: point in plane
        :type point: ndarray
        :returns: parameterization of plane
        :rtype: ndarray
        """
        return np.array((*vec, np.dot(vec, point)))

    @staticmethod
    def cell2planes(cell, pbc):
        """Get the parameterization of the sizes of a ase.Atom cell

        :param cell: ase.Atom cell
        :type cell: obj
        :param pbc: shift of boundaries to be used with periodic boundary condition
        :type pbc: float
        :returns: parameterization of cell plane sides
        :rtype: list of ndarray

        3 planes intersect the origin by ase design.
        """
        a = cell[0]
        b = cell[1]
        c = cell[2]

        n1 = np.cross(a, b)
        n2 = np.cross(c, a)
        n3 = np.cross(b, c)

        # n1 = n1/np.dot(n1, n1)
        # n2 = n2/np.dot(n2, n2)
        # n3 = n3/np.dot(n3, n3)

        origin = np.array([0, 0, 0]) + pbc / 2
        top = (a + b + c) - pbc / 2

        plane1 = Geometry.vec_and_point_to_plane(n1, origin)
        plane2 = Geometry.vec_and_point_to_plane(n2, origin)
        plane3 = Geometry.vec_and_point_to_plane(n3, origin)
        plane4 = Geometry.vec_and_point_to_plane(-n1, top)
        plane5 = Geometry.vec_and_point_to_plane(-n2, top)
        plane6 = Geometry.vec_and_point_to_plane(-n3, top)

        return [plane1, plane2, plane3, plane4, plane5, plane6]

    @staticmethod
    def extract_box_properties(center, length, lo_corner, hi_corner):
        """Given two of the properties 'center', 'length', 'lo_corner',
        'hi_corner', return all the properties. The properties that
        are not given are expected to be 'None'.
        """
        # exactly two arguments have to be non-none
        if sum(x is None for x in [center, length, lo_corner, hi_corner]) != 2:
            raise ValueError("Exactly two arguments have to be given")

        # declare arrays to allow mathematical operations
        center, length = np.asarray(center), np.asarray(length)
        lo_corner, hi_corner = np.asarray(lo_corner), np.asarray(hi_corner)
        relations = [["lo_corner",              "hi_corner - length",
                      "center - length / 2",    "2 * center - hi_corner"],
                     ["hi_corner",              "lo_corner + length",
                      "center + length / 2",    "2 * center - lo_corner"],
                     ["length / 2",             "(hi_corner - lo_corner) / 2",
                      "hi_corner - center",     "center - lo_corner"],
                     ["center",                 "(hi_corner + lo_corner) / 2",
                      "hi_corner - length / 2", "lo_corner + length / 2"]]

        # compute all relations
        relation_list = []
        for relation in relations:
            for i in relation:
                try:
                    relation_list.append(eval(i))
                except TypeError:
                    continue

        # keep the non-None relations
        for i, relation in enumerate(relation_list):
            if None in relation:
                del relation_list[i]
        return relation_list

    def packmol_structure(self, number, side):
        """Make structure to be used in PACKMOL input script

        :param number: number of water molecules
        :type number: int
        :param side: pack water inside/outside of geometry
        :type side: str
        :returns: string with information about the structure
        :rtype: str
        """
        structure = "structure water.pdb\n"
        structure += f"  number {number}\n"
        structure += f"  {side} {self.__repr__()} "
        for param in self.params:
            structure += f"{param} "
        structure += "\nend structure\n"
        return structure


class PlaneBoundTriclinicGeometry(Geometry):
    """Triclinic crystal geometry based on ase.Atom cell

    :param cell: ase.Atom cell
    :type cell: obj
    :param pbc: shift of boundaries to be used with periodic boundary condition
    :type pbc: float
    """
    def __init__(self, cell, pbc=0.0):
        self.planes = self.cell2planes(cell, pbc)
        self.cell = cell
        self.ll_corner = [0, 0, 0]
        a = cell[0, :]
        b = cell[1, :]
        c = cell[2, :]
        self.ur_corner = a + b + c

    def packmol_structure(self, number, side):
        """Make structure to be used in PACKMOL input script
        """
        if side == "inside":
            side = "over"
        elif side == "outside":
            side = "below"
        structure = "structure water.pdb\n"
        structure += f"  number {number}\n"
        for plane in self.planes:
            structure += f"  {side} plane "
            for param in plane:
                structure += f"{param} "
            structure += "\n"
        structure += "end structure\n"
        return structure

    def __call__(self, position):
        raise NotImplementedError


class SphereGeometry(Geometry):
    """Spherical geometry.

    :param center: Center of sphere
    :type center: array_like
    :param radius: radius of sphere
    :type length: float
    """

    def __init__(self, center, radius, **kwargs):
        super().__init__(**kwargs)
        self.center = center
        self.radius = radius
        self.radius_squared = radius**2
        self.params = list(self.center) + [radius]
        self.ll_corner = np.array(center) - radius
        self.ur_corner = np.array(center) + radius

    def __repr__(self):
        return 'sphere'

    def __call__(self, atoms):
        atoms.append(Atom(position=self.center))
        tmp_pbc = atoms.get_pbc()
        atoms.set_pbc(self.periodic_boundary_condition)
        distances = atoms.get_distances(-1, list(range(len(atoms)-1)),
                                        mic=self.minimum_image_convention)
        atoms.pop()
        atoms.set_pbc(tmp_pbc)
        indices = distances**2 < self.radius_squared
        return indices


class CubeGeometry(Geometry):
    """Cubic geometry.

    :param center: center of cube
    :type center: array_like
    :param length: length of each side
    :type length: float
    """

    def __init__(self, center, length, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.length_half = np.array(length) / 2
        self.center = np.array(center)
        self.ll_corner = self.center - self.length_half
        self.ur_corner = self.center + self.length_half
        self.params = list(self.ll_corner) + [self.length]

    def __repr__(self):
        return 'cube'

    def __call__(self, atoms):
        positions = atoms.get_positions()
        dist = self.distance_point_plane(np.eye(3), self.center, positions)
        indices = np.all((np.abs(dist) <= self.length_half), axis=1)
        return indices


class BoxGeometry(Geometry):
    """Box geometry.

    :param center: geometric center of box
    :type center: array_like
    :param length: length of box in all directions
    :type length: array_like
    :param lo_corner: lower corner
    :type lo_corner: array_like
    :param hi_corner: higher corner
    :type hi_corner: array_like
    """

    def __init__(self, center=None, length=None, lo_corner=None,
                 hi_corner=None, **kwargs):
        super().__init__(**kwargs)
        props = self.extract_box_properties(center, length, lo_corner, hi_corner)
        self.ll_corner, self.ur_corner, self.length_half, self.center = props
        self.params = list(self.ll_corner) + list(self.ur_corner)
        self.length = self.length_half*2

    def __repr__(self):
        return 'box'

    def __call__(self, atoms):
        positions = atoms.get_positions()
        dist = self.distance_point_plane(np.eye(3), self.center, positions)
        indices = np.all((np.abs(dist) <= self.length_half), axis=1)
        return indices

    def volume(self):
        return np.prod(self.length)


class BlockGeometry(Geometry):
    """This is a more flexible box geometry, where the angle

    :param center: the center point of the block
    :type center: array_like
    :param length: the spatial extent of the block in each direction.
    :type length: array_like
    :param orientation: orientation of block
    :type orientation: nested list / ndarray_like

    NB: Does not support pack_water and packmol
    NB: This geometry will be deprecated
    """

    def __init__(self, center, length, orientation=[], **kwargs):
        super().__init__(**kwargs)
        assert len(center) == len(length), \
            ("center and length need to have equal shapes")
        self.center = np.array(center)
        self.length = np.array(length) / 2

        # Set coordinate according to orientation
        if len(orientation) == 0:
            # orientation.append(np.random.randn(len(center)))
            orientation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if len(orientation) == 1:
            n_x = np.array(orientation[0])
            n_y = np.random.randn(len(center))
            n_y -= n_y.dot(n_x) * n_x
            orientation.append(n_y)
        if len(orientation) == 2:
            orientation.append(np.cross(orientation[0], orientation[1]))
        orientation = np.array(orientation, dtype=float)
        self.orientation = orientation / np.linalg.norm(orientation, axis=1)

    def __repr__(self):
        return 'block'

    def packmol_structure(self, number, side):
        """Make structure to be used in PACKMOL input script
        """
        raise NotImplementedError("BlockGeometry does not support pack_water")

    def __call__(self, atoms):
        tmp_pbc = atoms.get_pbc()
        atoms.set_pbc(self.periodic_boundary_condition)
        positions = atoms.get_positions()
        atoms.set_pbc(tmp_pbc)
        indices = np.all((np.abs(self.distance_point_plane(
            self.orientation, self.center, positions)) <= self.length), axis=1)
        return indices

class ConeGeometry(Geometry):
    """Geometry of a cone with its axis in the z direction"""

    def __init__(self, point, height, radius, **kwargs):
        super().__init__(**kwargs)
        self.point = np.asarray(point)
        self.radius = radius
        self.height = height
    
    def __call__(self, atoms):
        positions = atoms.get_positions()
        dist = self.distance_point_line(np.array([0,0,1]), self.point, positions)
        indices = dist < (self.radius - self.radius*(positions[:,2]-self.point[2])/self.height)
        return indices


class PlaneGeometry(Geometry):
    """Remove all particles on one side of one or more planes. Can be used to
    form any 3d polygon, among other geometries

    :param point: point on plane
    :type point: array_like
    :param normal: vector normal to plane
    :type normal: array_like
    """

    def __init__(self, point, normal, **kwargs):
        super().__init__(**kwargs)
        assert len(point) == len(normal), \
            "Number of given points and normal vectors have to be equal"

        self.point = np.atleast_2d(point)
        normal = np.atleast_2d(normal)
        self.normal = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]

    def packmol_structure(self, number, side):
        """Make structure to be used in PACKMOL input script
        """
        if side == "inside":
            side = "over"
        elif side == "outside":
            side = "below"

        ds = np.einsum('ij,ij->j', self.point, self.normal)

        structure = "structure water.pdb\n"
        structure += f"  number {number}\n"
        for plane in range(len(self.normal)):
            a, b, c = self.normal[side]
            d = ds[side]
            structure += f"  {side} plane {a} {b} {c} {d} \n"
        structure += "end structure\n"
        return structure

    def __call__(self, atoms):
        positions = atoms.get_positions()
        dist = self.point[:, np.newaxis] - positions
        indices = np.all(np.einsum('ijk,ik->ij', dist, self.normal) > 0, axis=0)
        return indices


class CylinderGeometry(Geometry):
    """Cylinder object.

    :param center: the center point of the cylinder
    :type center: array_like
    :param radius: cylinder radius
    :type radius: float
    :param length: cylinder length
    :type length: float
    :param orientation: orientation of cylinder, given as a vector pointing
                        along the cylinder. Pointing in x-direction by default.
    :type orientation: array_like
    """

    def __init__(self, center, radius, length, orientation=None, **kwargs):
        super().__init__(**kwargs)
        self.center = np.array(center)
        self.radius = radius
        self.length_half = length / 2
        if orientation is None:
            self.orientation = np.zeros_like(center)
            self.orientation[0] = 1
        else:
            orientation = np.array(orientation, dtype=float)
            self.orientation = orientation / np.linalg.norm(orientation)
        self.params = list(center) + list(self.orientation) + [radius, length]

    def __repr__(self):
        return 'cylinder'

    def __call__(self, atoms):
        positions = atoms.get_positions()
        dist_inp = (self.orientation, self.center, positions)
        dist_line = self.distance_point_line(*dist_inp)
        dist_plane = self.distance_point_plane(*dist_inp).flatten()
        indices = (dist_line <= self.radius) & (dist_plane <= self.length_half)
        return indices


class BerkovichGeometry(Geometry):
    # TODO: Implement support for packmol through plane geometry
    def __init__(self, tip, axis=[0, 0, -1], angle=np.radians(65.27)):
        self.indenter_angle = angle
        self.tip = np.asarray(tip)
        self.axis = np.asarray(axis)
        self.plane_directions = []
        self._create_plane_directions()

    def _create_plane_directions(self):
        xy_angles = [0, np.radians(120), np.radians(240)]
        for xy_angle in xy_angles:
            z_component = np.cos(np.pi / 2 - self.indenter_angle)
            xy_component = np.sin(np.pi / 2 - self.indenter_angle)
            self.plane_directions.append(np.asarray([
                xy_component * np.cos(xy_angle),
                xy_component * np.sin(xy_angle),
                z_component
            ]))

    def packmol_structure(self, number, side):
        """Make structure to be used in PACKMOL input script
        """
        raise NotImplementedError(
            "BerkovichGeometry is not yet supported by pack_water")

    def __call__(self, atoms):
        positions = atoms.get_positions()
        rel_pos = positions-self.tip
        is_inside_candidate1 = np.dot(rel_pos, self.plane_directions[0]) > 0
        is_inside_candidate2 = np.dot(rel_pos, self.plane_directions[1]) > 0
        is_inside_candidate3 = np.dot(rel_pos, self.plane_directions[2]) > 0
        is_inside = np.logical_and(np.logical_and(
            is_inside_candidate1, is_inside_candidate2), is_inside_candidate3)
        return is_inside


class EllipsoidGeometry(Geometry):
    """Ellipsoid geometry, satisfies the equation

    (x - x0)^2   (y - y0)^2   (z - z0)^2
    ---------- + ---------- + ---------- = d
        a^2          b^2          c^2

    :param center: center of ellipsoid (x0, y0, z0)
    :type center: array_like
    :param length_axes: length of each axis (a, b, c)
    :type length_axes: array_like
    :param d: scaling
    :type d: float
    """

    # TODO: Add orientation argument

    def __init__(self, center, length_axes, d, **kwargs):
        super().__init__(**kwargs)
        self.center = np.asarray(center)
        self.axes_sqrd = np.asarray(length_axes)**2
        self.d = d
        self.params = list(self.center) + list(self.length) + [self.d]
        self.ll_corner = self.center - self.length
        self.ur_corner = self.center + self.length

    def __repr__(self):
        return 'ellipsoid'

    def __call__(self, atoms):
        positions = atoms.get_positions()
        positions_shifted_sqrd = (positions - self.center)**2
        LHS = np.sum(positions_shifted_sqrd / self.axes_sqrd, axis=1)
        indices = (LHS <= self.d)
        return indices


class EllipticalCylinderGeometry(Geometry):
    """Elliptical Cylinder

    :param center: center of elliptical cylinder
    :type center: array_like
    :param a: axes along x-axis
    :type a: float
    :param b: axes along y-axis
    :type b: float
    :param length: length of cylinder
    :type length: float
    :param orientation: which way the cylinder should point
    :type orientation: ndarray

    NB: This geometry is not supported by packmol or pack_water
    """

    # TODO: Fix orientation argument (two separate orientations)

    def __init__(self, center, a, b, length, orientation=None, **kwargs):
        super().__init__(**kwargs)
        self.center = np.asarray(center)
        self.axes_sqrd = np.asarray([a**2, b**2])
        self.length_half = np.asarray(length) / 2

        if orientation is None:
            self.orientation = np.zeros_like(center)
            self.orientation[0] = 1
        else:
            orientation = np.array(orientation, dtype=float)
            self.orientation = orientation / np.linalg.norm(orientation)

    def packmol_structure(self, number, side):
        """Make structure to be used in PACKMOL input script
        """
        raise NotImplementedError(
            "EllipticalCylinderGeometry is not supported by pack_water")

    def __call__(self, atoms):
        positions = atoms.get_positions()
        positions_shifted_sqrd = (positions - self.center)**2
        dist_inp = (self.orientation, self.center, positions)
        dist_plane = self.distance_point_plane(*dist_inp).flatten()
        ellipse = np.sum(positions_shifted_sqrd / self.axes_sqrd, axis=1)
        indices = (dist_plane <= self.length_half) & (ellipse <= 1)
        return indices


class ProceduralSurfaceGeometry(Geometry):
    """Creates procedural noise on a surface defined by a point, a normal
    vector and a thickness.

    :param point: an equilibrium point of noisy surface
    :type point: array_like
    :param normal: normal vector of noisy surface, surface is carved out in the poiting direction
    :type normal: array_like
    :param thickness: thickness of noise area
    :type thickness: float
    :param scale: scale of noise structures
    :type scale: int
    :param method: noise method, either 'simplex' or 'perlin'
    :type method: str
    :param f: arbitrary R^2 => R function to be added to the noise
    :type f: func
    :param threshold: define a threshold to define two-level surface by noise
    :type threshold: float
    :param repeat: define at what lengths the noise should repeat, default is surface length (if repeat=True)
    :type repeat: array_like or bool or float
    :param angle: angle of triclinic surface given in degrees
    :type angle: float
    :param seed: seed used in procedural noise
    :type seed: int
    """

    def __init__(self, point, normal, thickness, scale=1, method='perlin',
                 f=lambda x, y: 0, threshold=None, repeat=False, angle=90,
                 seed=45617, **kwargs):
        assert len(point) == len(normal), \
            "Number of given points and normal vectors have to be equal"

        if method == "simplex":
            self.noise = snoise2
        elif method == "perlin":
            self.noise = pnoise2

        self.point = np.atleast_2d(point)
        normal = np.atleast_2d(normal)
        self.normal = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]
        self.thickness = thickness
        self.scale = int(scale)
        self.repeat = repeat
        self.f = f
        self.threshold = threshold
        self.angle = angle

        if repeat:
            kwargs['repeatx'], kwargs['repeaty'] = self.scale, self.scale
        kwargs['base'] = seed
        self.kwargs = kwargs

    def packmol_structure(self, number, side):
        """Make structure to be used in PACKMOL input script
        """
        raise NotImplementedError(
            "ProceduralNoiseSurface is not supported by pack_water")

    def __call__(self, atoms):
        positions = atoms.get_positions()
        # calculate distance from particles to the plane defined by
        # the normal vector and the point
        dist = self.distance_point_plane(self.normal, self.point, positions)
        # find the points on plane
        point_plane = positions + np.einsum('ij,kl->jkl', dist, self.normal)
        # coordinate indices
        ind = [0, 1, 2]
        # a loop is actually faster than an all-numpy implementation
        # since pnoise3/snoise3 are written in C++
        noises = np.empty(dist.shape)
        for i, normal in enumerate(self.normal):
            # choose indices to be used in coordinate transformation
            k, l = np.delete(ind, np.argmax(normal))
            # transform space coordinates onto structure surface
            xs = point_plane[i, :, k]*(1-normal[k]**2)**(-1/2)
            ys = point_plane[i, :, l]*(1-normal[l]**2)**(-1/2)
            if isinstance(self.repeat, bool):
                # find length of surface
                lx = np.max(xs) - np.min(xs)
                ly = np.max(ys) - np.min(ys)
            elif hasattr(self.repeat, "__len__"):
                # self.repeat is array-like
                lx, ly = self.repeat
            else:
                # self.repeat is float
                lx = ly = self.repeat
            # scale coordinates with length of surface
            xs_scale = self.scale * xs/lx
            ys_scale = self.scale * ys/ly
            # transform from orthorhombic cell to triclinic cell
            xs_scale += ys_scale * np.cos(np.deg2rad(self.angle))
            for j, point in enumerate(point_plane[i]):
                # add function values to noise array
                noises[j] = - self.f(xs[j], ys[j])
                # add noise value to noise array
                noise_val = self.noise(xs_scale[j], ys_scale[j], **self.kwargs)
                if self.threshold is None:
                    noises[j] += (noise_val + 1) / 2
                else:
                    noises[j] += noise_val > self.threshold
        # find distance from particles to noisy surface
        dist = np.einsum('ijk,ik->ij', self.point[:, np.newaxis] - positions,
                         self.normal)
        noises = noises.flatten() * self.thickness
        indices = np.all(dist < noises, axis=0)
        return ~indices

class OctahedronGeometry(PlaneGeometry):
    """A rectangular octahedron geometry to be used for silicon carbide (SiC)
    All sides are assumed to have a normal vector pointing where are components
    have the same magnitude (ex. (1, 1, 1))

    :param d: (shortest) length from octahedron center to sides
    :type d: float
    :param center: center of octahedron
    :type center: array_like
    """
    def __init__(self, d, center=[0, 0, 0]):
        # make list of normal vectors
        bin_list = []
        for i in range(8):
            binary = format(i, '#05b')
            bin_list.append(list(binary[2:]))
        normals = np.asarray(bin_list, dtype=int)
        normals[normals == 0] = -1
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

        # find points in planes
        points = d * normals + np.asarray(center)
        super().__init__(points, normals)


class DodecahedronGeometry(PlaneGeometry):
    """A convex rectangular dodecahedron geometry to be used for silicon
    carbide (SiC).

    :param d: (shortest) length from dodecahedron center to sides
    :type d: float
    :param center: center of dodecahedron
    :type center: array_like
    """
    def __init__(self, d, center=[0, 0, 0]):
        # make list of normal vectors
        lst = [[+1, +1, 0], [+1, 0, +1], [0, +1, +1], [+1, -1, 0],
               [+1, 0, -1], [+0, 1, -1], [-1, +1, 0], [-1, 0, +1],
               [0, -1, +1], [-1, -1, 0], [-1, 0, -1], [0, -1, -1]]
        normals = np.asarray(lst, dtype=int)
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

        # find points in planes
        points = d * normals + np.asarray(center)
        super().__init__(points, normals)

class NotchGeometry(Geometry):
    """Carve out a notch geometry in a structure

    :param entry: The starting poing of the crack
    :type entry: array_like
    :param vector_in: The length of the crack
    :type vector_in: array_like
    :param vector_up: The thickness of the crack above and below the starting point
    :type vector_up: array_like
    """

    def __init__(self, entry, vector_in, vector_up):
        self.entry = np.asarray(entry)
        self.vector_in = np.asarray(vector_in)
        self.vector_up = np.asarray(vector_up)
        self.tip = self.entry+self.vector_in

        p1 = self.entry + self.vector_up
        p2 = self.entry + self.vector_in
        p3 = p2 + np.cross(self.vector_in, self.vector_up)
        self.normal_upper = np.cross(p3-p1, p2-p1)

        p1 = self.entry - self.vector_up
        p2 = self.entry + self.vector_in
        p3 = p2 + np.cross(self.vector_in, -self.vector_up)
        self.normal_lower  = np.cross(p3-p1, p2-p1)

    def __repr__(self):
        return 'crack'

    def __call__(self, atoms):
        position = atoms.get_positions()
        dist = self.entry-position
        is_inside1 = np.dot(dist, self.vector_in) > 0
        dist = self.tip-position
        is_inside2 = np.dot(dist, self.normal_upper) < 0
        is_inside3 = np.dot(dist, self.normal_lower) < 0

        indicies = np.logical_not(np.logical_and(np.logical_not(is_inside1), np.logical_or(is_inside2, is_inside3)))

        return indicies


class ProceduralSurfaceGridGeometry(Geometry):
    """Creates tileable procedural noise on a surface defined by a grid and a
    normal vector. Noise is applied throughout the direction of the normal.

    :param normal: normal vector of noisy surface, surface is carved out
                   in the poiting direction
    :type normal: array_like
    :param scale: scale of noise structures
    :type scale: float
    :param grid: Number of grid cells in each direction perpendicular to the
                 normal vector.
    :type grid: array_like
    :param threshold: Threshold for Simplex values to create a two-level
                      surface.
    :type threshold: float
    :param seed: Seed for procedural noise.
    :type seed: int
    :param period: Period for randomization of Simplex permutation matrix.
    :type period: int
    :returns: ndarray of bools stating which atoms to remove.
    :rtype: ndarray
    """

    def __init__(self, normal, scale=10, threshold=0, seed=1,
                 grid=(50, 100), period=4096, **kwargs):
        assert len(grid) == 2, \
            "Method only supports two-dimensional grid structure"
        assert (np.asarray(normal) == 0).sum() == 2, \
            "Implementation of grid only supports surface normal in one dimension"
        assert len(np.asarray(normal).shape) == 1, \
            "Only single surface normal is supported"
        if seed == 0:
            warnings.warn(
                "Seed 0 and 1 produces the same noise")

        # Randomize permutation matrix for Simplex noise
        randomize(seed=seed, period=period)

        normal = np.atleast_2d(normal)
        self.normal = normal / np.linalg.norm(normal, axis=1)
        self.noise = snoise2r
        self.scale = scale
        self.threshold = threshold
        self.n1, self.n2 = grid
        self.kwargs = kwargs

        # Noise grid can be used to create images by accessing
        # geometry.noise_grid after carving
        self.noise_grid = np.zeros((self.n1, self.n2), dtype=int)

    def packmol_structure(self, number, side):
        """Make structure to be used in PACKMOL input script
        """
        raise NotImplementedError(
            "ProceduralSurfaceGridGeometry is not supported by pack_water")

    def __call__(self, atoms):
        positions = atoms.get_positions()
        lens = atoms.cell.cellpar()[:3]

        ind = [0, 1, 2]
        k, l = np.delete(ind, np.argmax(self.normal))

        lx, ly = lens[k], lens[l]

        # Create grid lengths
        gridx = np.linspace(0, lx, self.n1)
        gridy = np.linspace(0, ly, self.n2)

        self.kwargs['repeatx'] = lx / self.scale
        self.kwargs['repeaty'] = ly / self.scale

        # Nested double for loop to create noise values for all points on the
        # grid
        noise_vals = np.array([
            self.noise(x / self.scale, y / self.scale, **self.kwargs)
            for x in gridx for y in gridy
        ]).reshape(self.n1, self.n2)

        # Set values to two-step by threshold
        self.noise_grid += noise_vals > self.threshold

        x = positions[:, k]
        y = positions[:, l]

        # (1 / grid cell lengths) for faster computations for x_i and y_i below
        gcell_lenx_inv = self.n1 / lx
        gcell_leny_inv = self.n2 / ly

        # Mapping positions to grid. Pairs x_i and y_i gives position of
        # particle on grid
        x_i = (positions[:, k] * gcell_lenx_inv).astype(int)
        y_i = (positions[:, l] * gcell_leny_inv).astype(int)

        # Assign particles to grid cells
        noises = self.noise_grid[x_i, y_i]

        indices = np.logical_not(noises.flatten())

        return indices

import ase
import numpy as np
import scipy
from itertools import product

from ocdata.core import Adsorbate, Surface
from ase.data import atomic_numbers, covalent_radii


class Adslab:
    """
    Initializes an adsorbate-catalyst system for a given Adsorbate and Surface.

    Arguments
    ---------
    surface: Surface
        Surface object.
    adsorbate: Adsorbate
        Adsorbate object.
    num_sites: int
        Number of sites to sample.
    num_augmentations_per_site: int
        Number of augmentations of the adsorbate per site. Total number of
        generated structures will be `num_sites` * `num_augmentations_per_site`.
    added_z: int
        Distance in Angstroms to add to the z-coordinate of the surface atoms
        above the site for placing the adsorbate.
    """

    def __init__(
        self,
        surface: Surface,
        adsorbate: Adsorbate,
        num_sites: int = 100,
        num_augmentations_per_site: int = 1,
        added_z: int = 2,
    ):
        self.surface = surface
        self.adsorbate = adsorbate
        self.num_sites = num_sites
        self.num_augmentations_per_site = num_augmentations_per_site

        self.sites = self.get_binding_sites(num_sites, added_z)
        self.structures = self.place_adsorbate_on_sites(
            self.sites, num_augmentations_per_site
        )
        self.structures = self.filter_unreasonable_structures(self.structures)

    def get_binding_sites(self, num_sites: int, added_z: int = 2):
        """
        Returns `num_sites` sites given the surface atoms' positions.
        """
        surface_atoms_idx = [
            i for i, atom in enumerate(self.surface.atoms) if atom.tag == 1
        ]
        surface_atoms_pos = self.surface.atoms[surface_atoms_idx].positions
        surface_atoms_elements = self.surface.atoms[
            surface_atoms_idx
        ].get_chemical_symbols()

        dt = scipy.spatial.Delaunay(surface_atoms_pos[:, :2])
        simplices = dt.simplices
        num_sites_per_triangle = int(np.ceil(num_sites / len(simplices)))
        all_sites = []
        for tri in simplices:
            triangle_positions = surface_atoms_pos[tri]
            triangle_els = [
                el for idx, el in enumerate(surface_atoms_elements) if idx in tri
            ]
            sites = get_random_sites_on_triangle(
                triangle_positions, triangle_els, num_sites_per_triangle
            )  # Comment(@brookwander): - dont like this but I also dont like the alternatives - open to options!
            all_sites += sites
        np.random.shuffle(all_sites)
        return all_sites[:num_sites]

    def place_adsorbate_on_site(self, site_info: np.ndarray):
        """
        Place the adsorbate at the given binding site.
        """
        simplex_elements, simplex_vertices, site = site_info
        adsorbate_c = self.adsorbate.atoms.copy()
        surface_c = self.surface.atoms.copy()

        # Rotate adsorbate along x, y, z.
        angles = np.random.uniform(0, 360, 3)
        adsorbate_c.rotate(angles[0], v="x", center="COM")
        adsorbate_c.rotate(angles[1], v="y", center="COM")
        adsorbate_c.rotate(angles[2], v="z", center="COM")

        # Translate adsorbate to binding site.
        com = adsorbate_c.get_center_of_mass()
        translation_vector = site - com
        adsorbate_c.translate(translation_vector)

        # Translate the adsorbate by the scaled normal so it is proximate to the surface
        scaled_normal = self._get_scaled_normal(
            adsorbate_c, simplex_vertices, simplex_elements
        )
        adsorbate_c.translate(scaled_normal)

        # Combine adsorbate and surface, and set tags correctly
        structure = surface_c + adsorbate_c
        tags = [2] * len(adsorbate_c)
        final_tags = list(surface_c.get_tags()) + tags
        structure.set_tags(final_tags)

        # Set pbc and cell.
        structure.cell = surface_c.cell
        structure.pbc = [True, True, False]

        return (site, structure)

    def place_adsorbate_on_sites(
        self, sites: list, num_augmentations_per_site: int = 1
    ):
        """
        Place the adsorbate at the given binding sites.
        """
        structures = []
        for site in sites:
            for _ in range(num_augmentations_per_site):
                structures.append(self.place_adsorbate_on_site(site))

        return structures

    def filter_unreasonable_structures(self, structures: list):
        """
        Filter out unreasonable adsorbate-surface systems.
        """
        filtered_structures = []
        for i in structures:
            site, adslab = i
            if is_adslab_structure_reasonable(adslab):
                filtered_structures.append(i)
        return filtered_structures

    def _get_scaled_normal(
        self, adsorbate_atoms, simplex_vertices, simplex_elements, tol=0.1
    ):
        """
        Translate the adsorbate along the normal so that it is proximate to the surface:
            1. Find the adsorbate atom - surface atom with the smallest distance (- radii)
            2. Solve for the distance at which the spheres defined by the atomic radii
                of the atoms in (1) first intersect. This will be done with the pythagorean
                theorem, where we construct a right triangle with these sides:
                    hypotenuse = radius_surface_atom + radius_adsorbate_atom
                    adjacent = distance from adsorbate coordinate projected onto the plane to
                        the surface atoms
                    placement_distance = distance we want to get!
            3. Calculate the scaled norm to be used for precise placement
        ** Assumes that (1) is equivalent to finding the pair that would intersect first via translation.
        This should be considered carefully. I think it should be true though.

        Args:
            adsorbate_atoms (np.ndarray): the vector normal to the plane defined by
                the 3 surface atoms along which the placement was calculated
            structure (ase.atoms.Atoms): the adsorbate surface configuration to be corrected.
            tol (float): [Angstroms] cushion added on to the intersection distance.

        Returns:
            (ase.atoms.Atoms): The atoms with corrected adsorbate positions.
        """
        # (1)

        ## temporarily give the adsorbate coordinates about the site, but far away
        normal_vector = find_normal_to_plane(simplex_vertices)
        scaled_normal_temporary = (
            normal_vector * 5 / np.sqrt(normal_vector.dot(normal_vector))
        )
        adsorbate_c2 = adsorbate_atoms.copy()
        adsorbate_c2.translate(scaled_normal_temporary)

        ## See which atoms are closest
        closest_combo, hypotenuse = self._find_closest_combo_and_min_distance(
            simplex_vertices, simplex_elements, adsorbate_c2
        )

        # (2)
        ## unpack coordinates to use
        adsorbate_atom_position = adsorbate_c2.get_positions()[closest_combo[1]]
        surface_atom_position = simplex_vertices[closest_combo[0]]

        ## get the length of "adjacent"
        v_ = adsorbate_atom_position - surface_atom_position
        projected_point = surface_atom_position + (
            v_
            - (np.dot(v_, normal_vector) / np.linalg.norm(normal_vector) ** 2)
            * normal_vector
        )

        adjacent = np.linalg.norm(projected_point - surface_atom_position)

        ## calculate distance via pythagorean theorem
        placement_distance = np.sqrt(hypotenuse**2 - adjacent**2) + tol

        # (3)
        scaled_vector = normal_vector * placement_distance / 5
        return scaled_vector

    def _find_closest_combo_and_min_distance(
        self, simplex_vertices, simplex_elements, adsorbate_atoms
    ):
        """
        Find the pair of surface and adsorbate atoms that are closest to one another.
        Calculate the min distance between them (sum of atomic radii)

        Args:
            simplex_vertices (np.ndarray): the coordinated of each of the surface
                atoms that define the placement simplex.
            simplex_elements (np.ndarray): the chemical symbols of each of the
                atoms in the simplex.
            adsorbate_atoms (ase.atoms.Atoms): the adsorbate atoms object that has
                been randomly placed far from the surface.

        Returns:
            (tuple): (simplex_idx, adsorbate_idx) of the most proximate atoms.
            (float): the minimum distance between the most proximate atoms.

        """
        adsorbate_coordinates = adsorbate_atoms.get_positions()
        adsorbate_elements = adsorbate_atoms.get_chemical_symbols()

        pairs = list(product(range(3), range(len(adsorbate_atoms))))
        post_radial_distances = []

        for combo in pairs:
            total_distance = np.linalg.norm(
                adsorbate_coordinates[combo[1]] - simplex_vertices[combo[0]]
            )
            post_radial_distance = (
                total_distance
                - covalent_radii[atomic_numbers[simplex_elements[combo[0]]]]
                - covalent_radii[atomic_numbers[adsorbate_elements[combo[1]]]]
            )
            post_radial_distances.append(post_radial_distance)

        closest_combo = pairs[post_radial_distances.index(min(post_radial_distances))]
        min_distance = (
            covalent_radii[atomic_numbers[simplex_elements[closest_combo[0]]]]
            + covalent_radii[atomic_numbers[adsorbate_elements[closest_combo[1]]]]
        )
        return closest_combo, min_distance


def get_random_sites_on_triangle(
    vertices: np.ndarray,
    elements: np.ndarray,
    num_sites: int = 10,
):
    """
    Sample `num_sites` random sites uniformly on a given 3D triangle.
    Following Sec. 4.2 from https://www.cs.princeton.edu/~funk/tog02.pdf.
    """
    assert len(vertices) == 3
    r1_sqrt = np.sqrt(np.random.uniform(0, 1, num_sites))[:, np.newaxis]
    r2 = np.random.uniform(0, 1, num_sites)[:, np.newaxis]
    sites = (
        (1 - r1_sqrt) * vertices[0]
        + r1_sqrt * (1 - r2) * vertices[1]
        + r1_sqrt * r2 * vertices[2]
    )
    return [(elements, vertices, i) for i in sites]


def is_adslab_structure_reasonable(
    adslab: ase.Atoms,
):
    """
    Check if the adsorbate-surface system is reasonable.
    """
    # Comment(@abhshkdz): This is a placeholder for now.
    return True


def find_normal_to_plane(vertices):
    """
    Get the normal vector to the plane defined by any 3 points.

    Args:
        vertices (list): the points about which to construct a plane

    Returns:
        (np.ndarray): normal vector
    """
    p1, p2, p3 = vertices
    v1 = p2 - p1
    v2 = p3 - p1
    return np.cross(v1, v2)

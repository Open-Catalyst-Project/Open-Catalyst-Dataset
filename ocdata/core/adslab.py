import ase
import numpy as np
import scipy
from itertools import product

from ocdata.core import Adsorbate, Surface
from ase.data import atomic_numbers, covalent_radii
import copy


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
    height_adjustment: float
        Distance in Angstroms to add to the height of placement incrementally
        until there is no atomic overlap.
    """

    def __init__(
        self,
        surface: Surface,
        adsorbate: Adsorbate,
        num_sites: int = 100,
        num_augmentations_per_site: int = 1,
        height_adjustment: float = 0.1,
    ):
        self.surface = surface
        self.adsorbate = adsorbate
        self.num_sites = num_sites
        self.num_augmentations_per_site = num_augmentations_per_site

        self.sites = self.get_binding_sites(num_sites)
        self.structures = self.place_adsorbate_on_sites(
            self.sites,
            num_augmentations_per_site,
            height_adjustment,
        )
        self.structures = self.filter_unreasonable_structures(self.structures)

    def get_binding_sites(self, num_sites: int):
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
                triangle_positions, num_sites_per_triangle
            )
            all_sites += sites
        np.random.shuffle(all_sites)
        return all_sites[:num_sites]

    def place_adsorbate_on_site(
        self,
        site: np.ndarray,
        height_adjustment: float = 0.1,
    ):
        """
        Place the adsorbate at the given binding site.
        """
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

        # Translate the adsorbate by the scaled normal until it is proximate to the surface
        self._reposition_adsorbate(adsorbate_c, height_adjustment)

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
        self,
        sites: list,
        num_augmentations_per_site: int = 1,
        height_adjustment: float = 0.1,
    ):
        """
        Place the adsorbate at the given binding sites.
        """
        structures = []
        for site in sites:
            for _ in range(num_augmentations_per_site):
                structures.append(self.place_adsorbate_on_site(site, height_adjustment))

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

    def _reposition_adsorbate(
        self,
        adsorbate_atoms: ase.atoms.Atoms,
        height_adjustment: float = 0.1,
    ):
        """
        Translate the adsorbate along the surface normal so that it is proximate
        to the surface but there is no atomic overlap by iteratively moving the
        adsorbate away from the surface, along its normal, until there is no
        more overlap.

        Args:
            adsorbate_atoms (ase.atoms.Atoms): the adsorbate atoms copy which
                is being manipulated during placement
            height_adjustment (float): [Agstroms] the added distance at each
                translation of the adsorbate away from the surface
        """
        surface_normal = self.surface.atoms.cell[2].copy()
        scaled_surface_normal = (
            surface_normal * height_adjustment / np.linalg.norm(surface_normal)
        )

        ## See which atoms are closest
        surface_atoms = self.surface.atoms[
            [idx for idx, tag in enumerate(self.surface.atoms.get_tags()) if tag == 1]
        ]
        surface_atoms_tiled = custom_tile_atoms(surface_atoms)

        total_distance_traversed = 0
        overlap_exists = there_is_overlap(adsorbate_atoms, surface_atoms_tiled)
        while overlap_exists:
            adsorbate_atoms.translate(scaled_surface_normal)
            overlap_exists = there_is_overlap(adsorbate_atoms, surface_atoms_tiled)


def get_random_sites_on_triangle(
    vertices: np.ndarray,
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
    return [i for i in sites]


def is_adslab_structure_reasonable(
    adslab: ase.Atoms,
):
    """
    Check if the adsorbate-surface system is reasonable.
    """
    # Comment(@abhshkdz): This is a placeholder for now.
    return True


def custom_tile_atoms(atoms: ase.atoms.Atoms):
    """
    Tile the atoms so that the center tile has the indices and positions of the
    untiled structure.

    Args:
        atoms (ase.atoms.Atoms): the atoms object to be tiled

    Return:
        (ase.atoms.Atoms): the tiled atoms which has been repeated 3 times in
            the x and y directions but maintains the original indices on the central
            unit cell.
    """
    vectors = [
        v for v in atoms.cell if ((round(v[0], 3) != 0) or (round(v[1], 3 != 0)))
    ]
    repeats = list(product([-1, 0, 1], repeat=2))
    repeats.remove((0, 0))
    new_atoms = copy.deepcopy(atoms)
    for repeat in repeats:
        atoms_shifted = copy.deepcopy(atoms)
        atoms_shifted.set_positions(
            atoms.get_positions() + vectors[0] * repeat[0] + vectors[1] * repeat[1]
        )
        new_atoms += atoms_shifted
    return new_atoms


def there_is_overlap(
    adsorbate_atoms: ase.atoms.Atoms, surface_atoms_tiled: ase.atoms.Atoms
):
    """
    Check to see if there is any atomic overlap between surface atoms
    and adsorbate atoms.

    Args:
        adsorbate_atoms (ase.atoms.Atoms): the adsorbate atoms copy which
            is being manipulated during placement
        surface_atoms_tiled: a tiled copy of the surface atoms from
            `custom_tile_atoms`

    Returns:
        (bool): True if there is atomic overlap, otherwise False
    """
    adsorbate_coordinates = adsorbate_atoms.get_positions()
    adsorbate_elements = adsorbate_atoms.get_chemical_symbols()

    surface_coordinates = surface_atoms_tiled.get_positions()
    surface_elements = surface_atoms_tiled.get_chemical_symbols()

    pairs = list(product(range(len(surface_atoms_tiled)), range(len(adsorbate_atoms))))
    unintersected_post_radial_distances = []

    for combo in pairs:
        total_distance = np.linalg.norm(
            adsorbate_coordinates[combo[1]] - surface_coordinates[combo[0]]
        )
        post_radial_distance = (
            total_distance
            - covalent_radii[atomic_numbers[surface_elements[combo[0]]]]
            - covalent_radii[atomic_numbers[adsorbate_elements[combo[1]]]]
        )
        unintersected_post_radial_distances.append(post_radial_distance >= 0)
    return not all(unintersected_post_radial_distances)

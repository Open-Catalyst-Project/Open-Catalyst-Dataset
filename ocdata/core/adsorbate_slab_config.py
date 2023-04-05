import copy
from itertools import product

import ase
import numpy as np
import scipy
from ase.data import atomic_numbers, covalent_radii
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor

from ocdata.core import Adsorbate, Slab
from ocdata.core.adsorbate import randomly_rotate_adsorbate
from scipy.optimize import fsolve


class AdsorbateSlabConfig:
    """
    Initializes an adsorbate-catalyst system for a given Adsorbate and Slab.

    Arguments
    ---------
    slab: Slab
        Slab object.
    adsorbate: Adsorbate
        Adsorbate object.
    num_sites: int
        Number of sites to sample.
    num_augmentations_per_site: int
        Number of augmentations of the adsorbate per site. Total number of
        generated structures will be `num_sites` * `num_augmentations_per_site`.
    adsorbate_height_step_size: float
        Distance in Angstroms to add to the height of placement incrementally
        until there is no atomic overlap.
    """

    def __init__(
        self,
        slab: Slab,
        adsorbate: Adsorbate,
        num_sites: int = 100,
        num_augmentations_per_site: int = 1,
        interstitial_gap: float = 0.1,
        mode: str = "random",
    ):
        assert mode in ["random", "heuristic"]

        self.slab = slab
        self.adsorbate = adsorbate
        self.num_sites = num_sites
        self.num_augmentations_per_site = num_augmentations_per_site
        self.interstitial_gap = interstitial_gap
        self.mode = mode

        self.sites = self.get_binding_sites(num_sites)
        self.atoms_list = self.place_adsorbate_on_sites(
            self.sites,
            num_augmentations_per_site,
            interstitial_gap,
        )
        # self.atoms_list = self.filter_unreasonable_structures(self.atoms_list)

    def get_binding_sites(self, num_sites: int):
        """
        Returns `num_sites` sites given the surface atoms' positions.
        """
        assert self.slab.has_surface_tagged()

        surface_atoms_idx = [
            i for i, atom in enumerate(self.slab.atoms) if atom.tag == 1
        ]
        surface_atoms_pos = self.slab.atoms[surface_atoms_idx].positions

        dt = scipy.spatial.Delaunay(surface_atoms_pos[:, :2])
        simplices = dt.simplices

        all_sites = []
        if self.mode == "random":
            num_sites_per_triangle = int(np.ceil(num_sites / len(simplices)))
            for tri in simplices:
                triangle_positions = surface_atoms_pos[tri]
                sites = get_random_sites_on_triangle(
                    triangle_positions, num_sites_per_triangle
                )
                all_sites += sites
            np.random.shuffle(all_sites)
        elif self.mode == "heuristic":
            ###
            # Comment(@abhshkdz): Initial attempt at not making the trip from
            # Ase to Pymatgen and back. But there are differences in the sites
            # returned by the two methods. So, we'll stick to Pymatgen for now.
            #
            # https://pymatgen.org/pymatgen.analysis.adsorption.html
            # Pymatgen assigns on-top, bridge, and hollow adsorption sites at
            # the nodes, edges, and face centers of the Delaunay triangulation.
            # https://github.com/materialsproject/pymatgen/blob/v2023.3.23/pymatgen/analysis/adsorption.py#L217
            # ontop: sites on top of surface atoms.
            # all_sites += [site for site in surface_atoms_pos]
            # for tri in simplices:
            #     # bridge: sites at centers of edges between surface atoms.
            #     for vpair in [[0, 1], [1, 2], [2, 0]]:
            #         all_sites.append(
            #             np.mean(surface_atoms_pos[tri][vpair], axis=0)
            #         )
            #     # hollow: sites at centers of Delaunay triangulation faces.
            #     # Note that we currently don't implement the check to avoid
            #     # sampling hollow sites in obtuse triangles:
            #     # https://github.com/materialsproject/pymatgen/blob/v2023.3.23/pymatgen/analysis/adsorption.py#L272
            #     all_sites.append(
            #         np.mean(surface_atoms_pos[tri], axis=0)
            #     )
            ###
            # Explicitly specify "surface" / "subsurface" atoms so that
            # Pymatgen doesn't recompute tags.
            site_properties = {"surface_properties": []}
            for atom in self.slab.atoms:
                if atom.tag == 1:
                    site_properties["surface_properties"].append("surface")
                else:
                    site_properties["surface_properties"].append("subsurface")
            struct = AseAtomsAdaptor.get_structure(self.slab.atoms)
            # Copy because Pymatgen doesn't let us update site_properties.
            struct = struct.copy(site_properties=site_properties)
            asf = AdsorbateSiteFinder(struct)
            # `distance` refers to the distance along the surface normal between
            # the slab and the adsorbate. We set it to 0 here since we later
            # explicitly check for atomic overlap and set the adsorbate height.
            all_sites += asf.find_adsorption_sites(distance=0)["all"]
            np.random.shuffle(all_sites)
        else:
            raise NotImplementedError

        return all_sites[:num_sites]

    def place_adsorbate_on_site(
        self,
        site: np.ndarray,
        interstitial_gap: float = 0.1,
    ):
        """
        Place the adsorbate at the given binding site.
        """
        adsorbate_c = self.adsorbate.atoms.copy()
        slab_c = self.slab.atoms.copy()

        # Rotate adsorbate along xyz, only if adsorbate has more than 1 atom.
        if len(self.adsorbate.atoms) > 1:
            adsorbate_c = randomly_rotate_adsorbate(
                adsorbate_c,
                mode=self.mode,
                binding_idx=self.adsorbate.binding_indices,
            )

        # Translate adsorbate to binding site.
        if self.mode == "random":
            placement_center = adsorbate_c.get_center_of_mass()
        elif self.mode == "heuristic":
            binding_idx = self.adsorbate.binding_indices[0]
            placement_center = adsorbate_c.positions[binding_idx]
        else:
            raise NotImplementedError
        translation_vector = site - placement_center
        adsorbate_c.translate(translation_vector)

        # Translate the adsorbate by the normal so it is far away
        normal = np.cross(self.slab.atoms.cell[0], self.slab.atoms.cell[1])
        unit_normal = normal / np.linalg.norm(normal)
        adsorbate_c2 = adsorbate_c.copy()
        adsorbate_c.translate(unit_normal * 5)

        adsorbate_slab_config = slab_c + adsorbate_c
        tags = [2] * len(adsorbate_c)
        final_tags = list(slab_c.get_tags()) + tags
        adsorbate_slab_config.set_tags(final_tags)

        scaled_normal = self._get_scaled_normal(
            adsorbate_slab_config,
            site,
            adsorbate_c,
            unit_normal,
            interstitial_gap,
        )
        adsorbate_c2.translate(scaled_normal * unit_normal)
        adsorbate_slab_config = slab_c + adsorbate_c2
        adsorbate_slab_config.set_tags(final_tags)

        # Set pbc and cell.
        adsorbate_slab_config.cell = slab_c.cell
        adsorbate_slab_config.pbc = [True, True, False]
        return adsorbate_slab_config

    def place_adsorbate_on_sites(
        self,
        sites: list,
        num_augmentations_per_site: int = 1,
        adsorbate_height_step_size: float = 0.1,
        interstitial_gap: float = 0.1,
    ):
        """
        Place the adsorbate at the given binding sites.
        """
        atoms_list = []
        for site in sites:
            for _ in range(num_augmentations_per_site):
                atoms_list.append(self.place_adsorbate_on_site(site, interstitial_gap))
        return atoms_list

    def _get_scaled_normal(
        self,
        adsorbate_slab_atoms: ase.Atoms,
        site: np.ndarray,
        adsorbate_atoms: ase.Atoms,
        unit_normal: np.ndarray,
        interstitial_gap: float = 0.1,
    ):
        """
        Translate the adsorbate along the surface normal so that it is proximate
        to the surface but there is no atomic overlap by explicitly solving for the
        point of intersection

        Args:
            adsorbate_slab_atoms (ase.Atoms): the initial adslab with poor placement
            site (np.ndarray): the coordinate of the site
            adsorbate_atoms (ase.Atoms): the translated adsorbate
            unit_normal (np.ndarray): the unit vector normal to the surface
            interstitial_gap (float): the desired distance between the covalent radii of the
                closest surface and adsorbate atom
        Returns:
            (float): the scaled normal vector for proper placement
        """

        atom_positions = adsorbate_slab_atoms.get_positions()

        # See which atoms are closest
        closest_idxs, d_min, surf_pos = self._find_closest_combo(adsorbate_slab_atoms)

        # Solve for the intersection
        if self.mode == "random":
            placement_center = adsorbate_atoms.get_center_of_mass()
        elif self.mode == "heuristic":
            binding_idx = self.adsorbate.binding_indices[0]
            placement_center = adsorbate_atoms.positions[binding_idx]
        u_ = atom_positions[closest_idxs[0]] - placement_center

        def fun(x):
            return (
                (surf_pos[0] - (site[0] + x * unit_normal[0] + u_[0])) ** 2
                + (surf_pos[1] - (site[1] + x * unit_normal[1] + u_[1])) ** 2
                + (surf_pos[2] - (site[2] + x * unit_normal[2] + u_[2])) ** 2
                - (d_min + interstitial_gap) ** 2
            )

        n_scale = fsolve(fun, d_min * 3)

        return n_scale[0]

    def _find_closest_combo(self, adsorbate_slab_atoms):
        """
        Find the pair of surface and adsorbate atoms that are closest to one another.
        Calculate the min distance between them (sum of atomic radii)
        Args:
            adsorbate_slab_atoms (ase.atoms.Atoms): the adsorbate-slab atoms configuration.
        Returns:
            (tuple): (surface_idx, adsorbate_idx) of the most proximate atoms.
            (float): the minimum distance between the most proximate atoms.
            (np.ndarray): coordinate of the most proximate surface atom with corrections for pbc
                if they are required.
        """
        adsorbate_idxs = [
            idx for idx, tag in enumerate(adsorbate_slab_atoms.get_tags()) if tag == 2
        ]
        surface_idxs = [
            idx for idx, tag in enumerate(adsorbate_slab_atoms.get_tags()) if tag == 1
        ]
        ads_slab_config_elements = adsorbate_slab_atoms.get_chemical_symbols()

        all_chemical_symbols = adsorbate_slab_atoms.get_chemical_symbols()

        pairs = list(product(adsorbate_idxs, surface_idxs))

        post_radial_distances = []

        for combo in pairs:
            total_distance = adsorbate_slab_atoms.get_distance(
                combo[0], combo[1], mic=True
            )
            post_radial_distance = (
                total_distance
                - covalent_radii[atomic_numbers[ads_slab_config_elements[combo[0]]]]
                - covalent_radii[atomic_numbers[ads_slab_config_elements[combo[1]]]]
            )
            post_radial_distances.append(post_radial_distance)

        closest_combo = pairs[post_radial_distances.index(min(post_radial_distances))]
        min_interstitial_distance = (
            covalent_radii[atomic_numbers[ads_slab_config_elements[closest_combo[0]]]]
            + covalent_radii[atomic_numbers[ads_slab_config_elements[closest_combo[1]]]]
        )

        min_distance = min(post_radial_distances) + min_interstitial_distance
        surface_pos = adsorbate_slab_atoms.get_positions()[closest_combo[1]]
        adsorbate_pos = adsorbate_slab_atoms.get_positions()[closest_combo[0]]

        # Get pbc corrected surface atom position if need be
        if (
            abs(
                adsorbate_slab_atoms.get_distance(closest_combo[0], closest_combo[1])
                - min_distance
            )
            > 0.1
        ):
            repeats = list(product([-1, 0, 1], repeat=2))
            repeats.remove((0, 0))
            for repeat in repeats:
                pbc_corrected_pos = (
                    self.slab.atoms.cell[0] * repeat[0]
                    + self.slab.atoms.cell[0] * repeat[1]
                    + surface_pos
                )
                if (
                    abs(
                        np.linalg.norm(pbc_corrected_pos - adsorbate_pos) - min_distance
                    )
                    < 0.1
                ):
                    break
            return closest_combo, min_interstitial_distance, pbc_corrected_pos
        else:
            return closest_combo, min_interstitial_distance, surface_pos


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


def custom_tile_atoms(atoms: ase.Atoms):
    """
    Tile the atoms so that the center tile has the indices and positions of the
    untiled structure.

    Args:
        atoms (ase.Atoms): the atoms object to be tiled

    Return:
        (ase.Atoms): the tiled atoms which has been repeated 3 times in
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


def there_is_overlap(adsorbate_atoms: ase.Atoms, surface_atoms_tiled: ase.Atoms):
    """
    Check to see if there is any atomic overlap between surface atoms
    and adsorbate atoms.

    Args:
        adsorbate_atoms (ase.Atoms): the adsorbate atoms copy which
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

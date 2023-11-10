import copy
import logging
import random
import warnings
from itertools import product
from typing import List

import ase
import numpy as np
import scipy
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import wrap_positions
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.optimize import fsolve

from ocdata.core import Adsorbate, AdsorbateSlabConfig, Slab
from ocdata.core.adsorbate import randomly_rotate_adsorbate


class MultipleAdsorbateSlabConfig(AdsorbateSlabConfig):
    def __init__(
        self,
        slab: Slab,
        adsorbates: List[Adsorbate],
        num_sites: int = 100,
        interstitial_gap: float = 0.1,
        mode: str = "random_site_heuristic_placement",
    ):
        assert mode in ["random", "heuristic", "random_site_heuristic_placement"]
        assert interstitial_gap < 5 and interstitial_gap >= 0

        self.slab = slab
        self.adsorbates = adsorbates
        self.num_sites = num_sites
        self.interstitial_gap = interstitial_gap
        self.mode = mode

        self.sites = self.get_binding_sites(num_sites)
        self.atoms, self.metadata_list = self.place_adsorbates_on_sites(
            self.sites,
            interstitial_gap,
        )

    def place_adsorbate_on_site(
        self,
        adsorbate: Adsorbate,
        site: np.ndarray,
        interstitial_gap: float = 0.1,
    ):
        """
        Place the adsorbate at the given binding site.

        This method is almost identical to the parent class. The only change
        here in this method is that the adsorbate is fed in as an argument
        rather than taken from the class attribute.
        """
        adsorbate_c = adsorbate.atoms.copy()
        slab_c = self.slab.atoms.copy()

        binding_idx = None
        if self.mode in ["heuristic", "random_site_heuristic_placement"]:
            binding_idx = np.random.choice(adsorbate.binding_indices)

        # Rotate adsorbate along xyz, only if adsorbate has more than 1 atom.
        sampled_angles = np.array([0, 0, 0])
        if len(adsorbate.atoms) > 1:
            adsorbate_c, sampled_angles = randomly_rotate_adsorbate(
                adsorbate_c,
                mode=self.mode,
                binding_idx=binding_idx,
            )

        # Translate adsorbate to binding site.
        if self.mode == "random":
            placement_center = adsorbate_c.get_center_of_mass()
        elif self.mode in ["heuristic", "random_site_heuristic_placement"]:
            placement_center = adsorbate_c.positions[binding_idx]
        else:
            raise NotImplementedError

        translation_vector = site - placement_center
        adsorbate_c.translate(translation_vector)

        # Translate the adsorbate by the normal so has no intersections
        normal = np.cross(self.slab.atoms.cell[0], self.slab.atoms.cell[1])
        unit_normal = normal / np.linalg.norm(normal)

        scaled_normal = self._get_scaled_normal(
            adsorbate_c,
            slab_c,
            site,
            unit_normal,
            interstitial_gap,
        )
        adsorbate_c.translate(scaled_normal * unit_normal)
        adsorbate_slab_config = slab_c + adsorbate_c
        tags = [2] * len(adsorbate_c)
        final_tags = list(slab_c.get_tags()) + tags
        adsorbate_slab_config.set_tags(final_tags)

        # Set pbc and cell.
        adsorbate_slab_config.cell = (
            slab_c.cell
        )  # Comment (@brookwander): I think this is unnecessary?
        adsorbate_slab_config.pbc = [True, True, False]

        return adsorbate_slab_config, sampled_angles

    def place_adsorbates_on_sites(
        self,
        sites: list,
        interstitial_gap: float = 0.1,
    ):
        """
        Place the adsorbate at the given binding sites.
        """
        # Build a fake atoms object with the positions as the sites.
        # This allows us to easily compute distances while accounting for perodicity.
        pseudo_atoms = Atoms(
            [1] * len(sites), positions=sites, cell=self.slab.atoms.get_cell(), pbc=True
        )
        num_sites = len(sites)

        ### Create metadata list
        metadata_list = []

        ### Build mapping to store distance of site to nearest adsorbate.
        ### Initialize to an arbitrarily large number to represent no adsorbates placed.
        distance_to_nearest_adsorbate_map = 1e10 * np.ones(num_sites)

        ### Randomly select a site to place the first adsorbate
        site_idx = np.random.choice(num_sites)
        site = sites[site_idx]

        initial_adsorbate = self.adsorbates[0]

        ### Place adsorbate on site
        base_atoms, sampled_angles = self.place_adsorbate_on_site(
            initial_adsorbate, site, interstitial_gap
        )

        metadata_list.append(
            {"adsorbate": initial_adsorbate, "site": site, "xyz_angles": sampled_angles}
        )

        ### For the initial adsorbate, update the distance mapping based
        distance_to_nearest_adsorbate_map = update_distance_map(
            distance_to_nearest_adsorbate_map,
            site_idx,
            initial_adsorbate,
            pseudo_atoms,
        )

        for idx, adsorbate in enumerate(self.adsorbates[1:]):
            binding_idx = adsorbate.binding_indices[0]
            binding_atom = adsorbate.atoms.get_atomic_numbers()[binding_idx]
            covalent_radius = covalent_radii[binding_atom]

            ### A site is allowed if the distance to the next closest adsorbate is
            ### at least the interstitial_gap + covalent radius of the binding atom away.
            ### The covalent radius of the nearest adsorbate is already considered in the
            ### distance mapping.
            mask = (
                distance_to_nearest_adsorbate_map >= interstitial_gap + covalent_radius
            ).nonzero()[0]

            site_idx = np.random.choice(mask)
            site = sites[site_idx]

            atoms, sampled_angles = self.place_adsorbate_on_site(
                adsorbate, site, interstitial_gap
            )

            ### Slabs are not altered in the adsorbat placement step
            ### We can add the adsorbate directly to the base atoms
            base_atoms += atoms[atoms.get_tags() == 2]

            distance_to_nearest_adsorbate_map = update_distance_map(
                distance_to_nearest_adsorbate_map,
                site_idx,
                adsorbate,
                pseudo_atoms,
            )

            metadata_list.append(
                {"adsorbate": adsorbate, "site": site, "xyz_angles": sampled_angles}
            )

        return base_atoms, metadata_list


def update_distance_map(prev_distance_map, site_idx, adsorbate, pseudo_atoms):
    """
    Given a new site and the adsorbate we plan on placing there,
    update the distance mapping to reflect the new distances from sites to nearest adsorbates.
    We incorporate the covalent radii of the placed adsorbate binding atom in our distance
    calculation to prevent atom overlap.
    """
    binding_idx = adsorbate.binding_indices[0]
    binding_atom = adsorbate.atoms.get_atomic_numbers()[binding_idx]
    covalent_radius = covalent_radii[binding_atom]

    new_site_distances = (
        pseudo_atoms.get_distances(site_idx, range(len(pseudo_atoms)), mic=True)
        - covalent_radius
    )

    ### update previous distance mapping by taking the minimum per-element distance between
    ### the new distance mapping for the placed site and the previous mapping.
    updated_distance_map = np.minimum(prev_distance_map, new_site_distances)

    return updated_distance_map

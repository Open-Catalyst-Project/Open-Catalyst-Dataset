import random

import numpy as np
import pytest
from ase.data import covalent_radii
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor

from ocdata.configs.paths import ADSORBATES_PKL_PATH, BULK_PKL_PATH
from ocdata.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab


@pytest.fixture(scope="class")
def load_data(request):
    request.cls.bulk = Bulk(bulk_id_from_db=0)
    request.cls.adsorbate = Adsorbate(adsorbate_id_from_db=80)


@pytest.mark.usefixtures("load_data")
class TestAdslab:
    def test_adslab_init(self):
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslab = AdsorbateSlabConfig(slab, self.adsorbate, num_sites=100)
        assert (
            len(adslab.atoms_list) == 100
        ), f"Insufficient number of structures. Expected 100, got {len(adslab.atoms_list)}"

        sites = ["%.04f_%.04f_%.04f" % (i[0], i[1], i[2]) for i in adslab.sites]
        assert (
            len(set(sites)) == 100
        ), f"Insufficient number of sites. Expected 100, got {len(set(sites))}"

        assert np.all(
            np.isclose(
                adslab.atoms_list[0].get_positions().mean(0),
                np.array([6.89636428, 6.64944068, 16.19283703]),
            )
        )
        assert np.all(
            np.isclose(
                adslab.atoms_list[1].get_positions().mean(0),
                np.array([7.04126794, 6.53481926, 16.14532353]),
            )
        )

    def test_num_augmentations_per_site(self):
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslab = AdsorbateSlabConfig(
            slab, self.adsorbate, num_sites=1, num_augmentations_per_site=100
        )
        assert len(adslab.atoms_list) == 100

        sites = ["%.04f_%.04f_%.04f" % (i[0], i[1], i[2]) for i in adslab.sites]
        assert len(set(sites)) == 1

    def test_adsorbate_height_step_size(self):
        """
        Test that the adsorbate atoms are all above `adsorbate_height_step_size`.

        Comment(@abhshkdz): This test is very loose and should be improved.
        All we're currently checking is that the adsorbate atoms are above
        `adsorbate_height_step_size` and not below it. A tighter version of the check would be
        to make sure that the gap between the adsorbate atoms and the surface
        is at least `adsorbate_height_step_size`.
        """
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslab = AdsorbateSlabConfig(
            slab, self.adsorbate, num_sites=100, adsorbate_height_step_size=2.0
        )
        assert len(adslab.atoms_list) == 100

        min_z = []
        for i in adslab.atoms_list:
            ads_idx = i.get_tags() == 2
            ads_pos = i.get_positions()[ads_idx]
            min_z.append(ads_pos[:, 2].min())

        assert np.all(np.array(min_z) > 2.0)

        adslab = AdsorbateSlabConfig(
            slab, self.adsorbate, num_sites=100, adsorbate_height_step_size=20.0
        )
        min_z = []
        for i in adslab.atoms_list:
            ads_idx = i.get_tags() == 2
            ads_pos = i.get_positions()[ads_idx]
            min_z.append(ads_pos[:, 2].min())

        assert np.all(np.array(min_z) > 20.0)

    def test_is_adsorbate_com_on_site(self):
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslab = AdsorbateSlabConfig(slab, self.adsorbate, num_sites=100, mode="random")

        sample_ids = np.random.randint(0, len(adslab.atoms_list), 10)
        for idx in sample_ids:
            site, atoms = adslab.sites[idx], adslab.atoms_list[idx]
            mask = atoms.get_tags() == 2
            adsorbate_atoms = atoms[mask]
            adsorbate_com = adsorbate_atoms.get_center_of_mass()
            # x,y coordinates should be the same
            assert np.isclose(site[:2], adsorbate_com[:2]).all()

    def test_is_adsorbate_binding_atom_on_site(self):
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslab = AdsorbateSlabConfig(
            slab, self.adsorbate, num_sites=100, mode="heuristic"
        )
        binding_idx = self.adsorbate.binding_indices[0]

        sample_ids = np.random.randint(0, len(adslab.atoms_list), 10)
        for idx in sample_ids:
            site, atoms = adslab.sites[idx], adslab.atoms_list[idx]
            mask = atoms.get_tags() == 2
            adsorbate_atoms = atoms[mask]
            binding_atom = adsorbate_atoms[binding_idx].position
            # x,y coordinates should be the same
            assert np.isclose(site[:2], binding_atom[:2]).all()

    def test_is_config_reasonable(self):
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslab = AdsorbateSlabConfig(slab, self.adsorbate, num_sites=100)

        samples = random.sample(adslab.atoms_list, 20)
        for atoms in samples:
            assert self.is_config_reasonable(atoms)

    def is_config_reasonable(self, atoms):
        adsorbate_mask = atoms.get_tags() == 2

        atomic_numbers = atoms.get_atomic_numbers()
        all_distances = atoms.get_all_distances(mic=True)

        adsorbate_atomic_numbers = atomic_numbers[adsorbate_mask]
        surface_atomic_numbers = atomic_numbers[~adsorbate_mask]
        pair_covalent_radii = (
            covalent_radii[adsorbate_atomic_numbers, None]
            + covalent_radii[surface_atomic_numbers]
        )

        adsorbate_distances = all_distances[adsorbate_mask][:, ~adsorbate_mask]
        pair_distances_minus_covalent = adsorbate_distances - pair_covalent_radii

        return not (pair_distances_minus_covalent < 0).any()

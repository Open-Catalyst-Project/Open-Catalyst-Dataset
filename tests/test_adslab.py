import random

import numpy as np
import pytest
from ase.data import covalent_radii

from ocdata.configs.paths import ADSORBATES_PKL_PATH, BULK_PKL_PATH
from ocdata.core import Adslab, Adsorbate, Bulk, Surface


@pytest.fixture(scope="class")
def load_data(request):
    request.cls.bulk = Bulk(bulk_id_from_db=0)
    request.cls.adsorbate = Adsorbate(adsorbate_id_from_db=80)


@pytest.mark.usefixtures("load_data")
class TestAdslab:
    def test_adslab_init(self):
        random.seed(1)
        np.random.seed(1)

        surface = Surface(self.bulk)
        adslab = Adslab(surface, self.adsorbate, num_sites=100)
        assert (
            len(adslab.structures) == 100
        ), f"Insufficient number of structures. Expected 100, got {len(adslab.structures)}"

        sites = [
            "%.04f_%.04f_%.04f" % (i[0][0], i[0][1], i[0][2]) for i in adslab.structures
        ]
        assert (
            len(set(sites)) == 100
        ), f"Insufficient number of sites. Expected 100, got {len(set(sites))}"

        assert np.all(
            np.isclose(
                adslab.structures[0][1].get_positions().mean(0),
                np.array([6.89636428, 6.64944068, 16.13494229]),
            )
        )
        assert np.all(
            np.isclose(
                adslab.structures[1][1].get_positions().mean(0),
                np.array([7.04126794, 6.53481926, 16.0874288]),
            )
        )

    def test_num_augmentations_per_site(self):
        random.seed(1)
        np.random.seed(1)

        surface = Surface(self.bulk)
        adslab = Adslab(
            surface, self.adsorbate, num_sites=1, num_augmentations_per_site=100
        )
        assert len(adslab.structures) == 100

        sites = [
            "%.04f_%.04f_%.04f" % (i[0][0], i[0][1], i[0][2]) for i in adslab.structures
        ]
        assert len(set(sites)) == 1

    def test_height_adjustment(self):
        """
        Test that the adsorbate atoms are all above `height_adjustment`.

        Comment(@abhshkdz): This test is very loose and should be improved.
        All we're currently checking is that the adsorbate atoms are above
        `height_adjustment` and not below it. A tighter version of the check would be
        to make sure that the gap between the adsorbate atoms and the surface
        is at least `height_adjustment`.
        """
        random.seed(1)
        np.random.seed(1)

        surface = Surface(self.bulk)
        adslab = Adslab(surface, self.adsorbate, num_sites=100, height_adjustment=2.0)
        assert len(adslab.structures) == 100

        min_z = []
        for i in adslab.structures:
            ads_idx = i[1].get_tags() == 2
            ads_pos = i[1].get_positions()[ads_idx]
            min_z.append(ads_pos[:, 2].min())

        assert np.all(np.array(min_z) > 2.0)

        adslab = Adslab(surface, self.adsorbate, num_sites=100, height_adjustment=20.0)
        min_z = []
        for i in adslab.structures:
            ads_idx = i[1].get_tags() == 2
            ads_pos = i[1].get_positions()[ads_idx]
            min_z.append(ads_pos[:, 2].min())

        assert np.all(np.array(min_z) > 20.0)

    def test_is_adsorbate_com_on_site(self):
        random.seed(0)

        surface = Surface(self.bulk)
        adslab = Adslab(surface, self.adsorbate, num_sites=100, mode="random")

        samples = random.sample(adslab.structures, 10)
        for sample in samples:
            site, atoms = sample
            mask = atoms.get_tags() == 2
            adsorbate_atoms = atoms[mask]
            adsorbate_com = adsorbate_atoms.get_center_of_mass()

            # x,y coordinates should be the same
            assert np.isclose(site[:2], adsorbate_com[:2]).all()

    def test_is_adsorbate_binding_idx_on_site(self):
        random.seed(0)

        surface = Surface(self.bulk)
        adslab = Adslab(surface, self.adsorbate, num_sites=100, mode="heuristic")
        binding_idx = self.adsorbate.binding_indices[0]

        samples = random.sample(adslab.structures, 10)
        for sample in samples:
            site, atoms = sample
            mask = atoms.get_tags() == 2
            adsorbate_atoms = atoms[mask]
            binding_atom = adsorbate_atoms[binding_idx]

            # x,y coordinates should be the same
            assert np.isclose(site[:2], binding_atom[:2]).all()

    def test_is_config_reasonable(self):
        random.seed(0)

        surface = Surface(self.bulk)
        adslab = Adslab(surface, self.adsorbate, num_sites=100)

        samples = random.sample(adslab.structures, 20)

        for sample in samples:
            site, atoms = sample
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

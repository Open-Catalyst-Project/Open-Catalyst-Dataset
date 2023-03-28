import random

import numpy as np
import pytest

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
        assert len(adslab.structures) == 100

        sites = [
            "%.04f_%.04f_%.04f" % (i[0][0], i[0][1], i[0][2]) for i in adslab.structures
        ]
        assert len(set(sites)) == 100

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

    def test_added_z(self):
        """
        Test that the adsorbate atoms are all above `added_z`.

        Comment(@abhshkdz): This test is very loose and should be improved.
        All we're currently checking is that the adsorbate atoms are above
        `added_z` and not below it. A tighter version of the check would be
        to make sure that the gap between the adsorbate atoms and the surface
        is at least `added_z`.
        """
        random.seed(1)
        np.random.seed(1)

        surface = Surface(self.bulk)
        adslab = Adslab(surface, self.adsorbate, num_sites=100, added_z=2.0)
        assert len(adslab.structures) == 100

        min_z = []
        for i in adslab.structures:
            ads_idx = i[1].get_tags() == 2
            ads_pos = i[1].get_positions()[ads_idx]
            min_z.append(ads_pos[:, 2].min())

        assert np.all(np.array(min_z) > 2.0)

        adslab = Adslab(surface, self.adsorbate, num_sites=100, added_z=20.0)
        min_z = []
        for i in adslab.structures:
            ads_idx = i[1].get_tags() == 2
            ads_pos = i[1].get_positions()[ads_idx]
            min_z.append(ads_pos[:, 2].min())

        assert np.all(np.array(min_z) > 20.0)
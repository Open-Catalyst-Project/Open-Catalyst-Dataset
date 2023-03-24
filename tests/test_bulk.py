import random

import numpy as np
import pytest

from ocdata.core import Bulk


@pytest.fixture(scope="class")
def load_bulk(request):
    request.cls.bulk = Bulk(bulk_id_from_db=0)


@pytest.mark.usefixtures("load_bulk")
class TestBulk:
    def test_bulk_init_from_id(self):
        bulk = Bulk(bulk_id_from_db=0)
        assert bulk.atoms.get_chemical_formula() == "Re2"

    def test_bulk_init_random(self):
        random.seed(1)
        np.random.seed(1)

        bulk = Bulk()
        assert bulk.atoms.get_chemical_formula() == "IrSn2"

    def test_surface_enumeration(self):
        precomputed_surfaces = self.bulk.get_precomputed_surfaces()
        assert len(precomputed_surfaces) == 15

        surfaces = self.bulk.compute_surfaces()
        assert len(surfaces) == 15

        assert [precomputed_surfaces[i] == surfaces[i] for i in range(15)]

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

    def test_slab_enumeration(self):
        precomputed_slabs = self.bulk.get_precomputed_slabs()
        assert len(precomputed_slabs) == 15

        slabs = self.bulk.compute_slabs()
        assert len(slabs) == 15

        assert [precomputed_slabs[i] == slabs[i] for i in range(15)]

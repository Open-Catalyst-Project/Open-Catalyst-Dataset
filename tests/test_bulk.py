import os
import pickle
import random

import numpy as np
import pytest

from ocdata.core import Bulk


@pytest.fixture(scope="class")
def load_bulk(request):
    cwd = os.getcwd()
    request.cls.idx = 0

    request.cls.precomputed_path = os.path.join(cwd, str(request.cls.idx) + ".pkl")
    request.cls.bulk = Bulk(bulk_id_from_db=request.cls.idx)


@pytest.mark.usefixtures("load_bulk")
class TestBulk:
    def test_bulk_init_from_id(self):
        bulk = Bulk(bulk_id_from_db=self.idx)
        assert bulk.atoms.get_chemical_formula() == "Re2"

    def test_bulk_init_random(self):
        random.seed(1)
        np.random.seed(1)

        bulk = Bulk()
        assert bulk.atoms.get_chemical_formula() == "IrSn2"

    def test_unique_slab_enumeration(self):
        slabs = self.bulk.get_slabs()

        seen = []
        for slab in slabs:
            assert slab not in seen
            seen.append(slab)

        # pymatgen-2023.1.20 + ase 3.22.1
        assert len(slabs) == 15

        with open(self.precomputed_path, "wb") as f:
            pickle.dump(slabs, f)

    def test_precomputed_slab(self):
        precomputed_slabs_dir = os.path.dirname(self.precomputed_path)

        precomputed_slabs = self.bulk.get_slabs(
            precomputed_slabs_dir=precomputed_slabs_dir
        )
        assert len(precomputed_slabs) == 15

        slabs = self.bulk.get_slabs()
        assert precomputed_slabs[0] == slabs[0]

        os.remove(self.precomputed_path)

    def test_slab_miller_enumeration(self):
        slabs_max_miller_1 = self.bulk.get_slabs(max_miller=1)
        assert self.get_max_miller(slabs_max_miller_1) == 1
        slabs_max_miller_2 = self.bulk.get_slabs(max_miller=2)
        assert self.get_max_miller(slabs_max_miller_2) == 2
        slabs_max_miller_3 = self.bulk.get_slabs(max_miller=3)
        assert self.get_max_miller(slabs_max_miller_3) == 3

    def get_max_miller(self, slabs):
        max_miller = 0
        for slab in slabs:
            millers = slab.millers
            max_miller = max(max_miller, max(millers))

        return max_miller

import random

import pytest

from ocdata.core import Adslab, Adsorbate, Bulk, Surface
from ocdata.utils.vasp import VASP_FLAGS, _clean_up_inputs


@pytest.fixture(scope="class")
def load_data(request):
    bulk_sample_1 = Bulk(bulk_id_from_db=24)
    surface_sample_1 = Surface(bulk_sample_1)
    adsorbate_sample_1 = Adsorbate(adsorbate_id_from_db=10)

    bulk_sample_2 = Bulk(bulk_id_from_db=100)
    surface_sample_2 = Surface(bulk_sample_2)
    adsorbate_sample_2 = Adsorbate(adsorbate_id_from_db=2)

    request.cls.adslab1 = Adslab(surface_sample_1, adsorbate_sample_1, num_sites=100)
    request.cls.adslab2 = Adslab(surface_sample_2, adsorbate_sample_2, num_sites=100)

    ALT_VASP_FLAGS = VASP_FLAGS.copy()
    ALT_VASP_FLAGS["nsw"] = 0
    ALT_VASP_FLAGS["laechg"] = False
    ALT_VASP_FLAGS["ncore"] = 1
    request.cls.alt_flags = ALT_VASP_FLAGS


@pytest.mark.usefixtures("load_data")
class TestVasp:
    def test_cleanup(self):
        atoms = self.adslab1.structures[0][1]
        atoms1, flags1 = _clean_up_inputs(atoms, VASP_FLAGS)

        # Check that kpts are computed and added to the flags
        assert "kpts" in flags1
        # Check that kpts weren't added to the original flags
        assert "kpts" not in VASP_FLAGS

        atoms2, flags2 = _clean_up_inputs(atoms, self.alt_flags)

        assert atoms1 == atoms2
        assert flags2 != flags1

    def test_unique_kpts(self):
        atoms1 = self.adslab1.structures[0][1]
        atoms2 = self.adslab2.structures[0][1]

        _, flags1 = _clean_up_inputs(atoms1, VASP_FLAGS)
        _, flags2 = _clean_up_inputs(atoms2, VASP_FLAGS)

        assert flags1 != flags2

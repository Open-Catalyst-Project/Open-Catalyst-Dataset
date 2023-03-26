import random

import numpy as np
import pytest

from ocdata.core import Adsorbate


class TestAdsorbate:
    def test_adsorbate_init_from_id(self):
        adsorbate = Adsorbate(adsorbate_id_from_db=0)
        assert adsorbate.atoms.get_chemical_formula() == "O"
        assert adsorbate.smiles == "*O"

    def test_adsorbate_init_random(self):
        random.seed(1)
        np.random.seed(1)

        adsorbate = Adsorbate()
        assert adsorbate.atoms.get_chemical_formula() == "C2H3O"
        assert adsorbate.smiles == "*COHCH2"
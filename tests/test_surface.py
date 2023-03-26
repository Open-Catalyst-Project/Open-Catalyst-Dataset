import random

import numpy as np

from ocdata.core import Bulk, Surface


class TestSurface:
    def test_surface_init_from_id(self):
        bulk = Bulk(bulk_id_from_db=0)
        slabs = bulk.get_slabs()

        surface = Surface(bulk, slabs[0])

        assert surface.atoms.get_chemical_formula() == "Re48"
        assert surface.millers == (1, 1, 1)
        assert surface.shift == 0.0

    def test_surface_init_random(self):
        random.seed(1)
        np.random.seed(1)

        bulk = Bulk(bulk_id_from_db=100)
        surface = Surface(bulk)

        assert surface.atoms.get_chemical_formula() == "Sn36"
        assert surface.millers == (1, 0, 0)
        assert surface.shift == 0.25

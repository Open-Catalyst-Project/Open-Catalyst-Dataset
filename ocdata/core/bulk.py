import os
import pickle

import ase
import numpy as np
import pymatgen
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ocdata.configs.constants import COVALENT_MATERIALS_MPIDS
from ocdata.configs.paths import BULK_PKL_PATH, PRECOMPUTED_SLABS_DIR_PATH
from ocdata.core.slab import Slab, compute_slabs


class Bulk:
    """
    Initializes a bulk object in one of 3 ways:
    - Directly pass in an ase.Atoms object.
    - Pass in index of bulk to select from bulk database.
    - Randomly sample a bulk from bulk database.

    Arguments
    ---------
    bulk_atoms: ase.Atoms
        Bulk structure.
    bulk_id_from_db: int
        Index of bulk to select if not doing a random sample.
    bulk_db_path: str
        Path to bulk database.
    precomputed_slabs_path: str
        Path to folder of precomputed slabs.
    """

    def __init__(
        self,
        bulk_atoms: ase.Atoms = None,
        bulk_id_from_db: int = None,
        bulk_db_path: str = BULK_PKL_PATH,
    ):
        self.bulk_id_from_db = bulk_id_from_db
        self.bulk_db_path = bulk_db_path

        if bulk_atoms is not None:
            self.atoms = bulk_atoms.copy()
            self.src_id = None
        elif bulk_id_from_db is not None:
            bulk_db = pickle.load(open(bulk_db_path, "rb"))
            bulk_obj = bulk_db[bulk_id_from_db]
            self.atoms, self.src_id = bulk_obj["atoms"], bulk_obj["src_id"]
        else:
            bulk_db = pickle.load(open(bulk_db_path, "rb"))
            bulk_id_from_db = np.random.randint(len(bulk_db))
            bulk_obj = bulk_db[bulk_id_from_db]
            self.atoms, self.src_id = bulk_obj["atoms"], bulk_obj["src_id"]

    def set_source_dataset_id(self, src_id: str):
        self.src_id = src_id

    def set_bulk_id_from_db(self, bulk_id_from_db: int):
        self.bulk_id_from_db = bulk_id_from_db

    def get_slabs(self, max_miller=2, precomputed_slabs_dir=None):
        """
        Returns a list of possible slabs for this bulk instance.
        """
        if precomputed_slabs_dir is not None and self.bulk_id_from_db is not None:
            slabs = Slab.from_precomputed_slabs_pkl(
                self,
                os.path.join(precomputed_slabs_dir, f"{self.bulk_id_from_db}.pkl"),
                max_miller=max_miller,
            )
        else:
            slabs = Slab.from_bulk_get_all_slabs(self, max_miller=max_miller)

        return slabs

    def __len__(self):
        return len(self.atoms)

    def __str__(self):
        return f"Bulk: ({self.atoms.get_chemical_formula()})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other) -> bool:
        return self.atoms == other.atoms

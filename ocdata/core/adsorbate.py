import os
import pickle

import ase
import numpy as np

from ocdata.configs.paths import ADSORBATES_PKL_PATH


class Adsorbate:
    """
    Initializes an adsorbate object in one of 3 ways:
    - Directly pass in an ase.Atoms object.
    - Pass in index of adsorbate to select from adsorbate database.
    - Randomly sample an adsorbate from adsorbate database.

    Arguments
    ---------
    adsorbate_atoms: ase.Atoms
        Adsorbate structure.
    adsorbate_id_from_db: int
        Index of adsorbate to select if not doing a random sample.
    adsorbate_db_path: str
        Path to adsorbate database.
    """

    def __init__(
        self,
        adsorbate_atoms: ase.Atoms = None,
        adsorbate_id_from_db: int = None,
        adsorbate_db_path: str = ADSORBATES_PKL_PATH,
    ):
        self.adsorbate_id_from_db = adsorbate_id_from_db
        self.adsorbate_db_path = adsorbate_db_path

        if adsorbate_atoms is not None:
            self.atoms = adsorbate_atoms.copy()
            self.smiles = None
            self.binding_indices = None
        elif adsorbate_id_from_db is not None:
            adsorbate_db = pickle.load(open(adsorbate_db_path, "rb"))
            self.atoms, self.smiles, self.binding_indices = adsorbate_db[
                adsorbate_id_from_db
            ]
        else:
            adsorbate_db = pickle.load(open(adsorbate_db_path, "rb"))
            adsorbate_id_from_db = np.random.randint(len(adsorbate_db))
            self.atoms, self.smiles, self.binding_indices = adsorbate_db[
                adsorbate_id_from_db
            ]
            self.adsorbate_id_from_db = adsorbate_id_from_db

    def __len__(self):
        return len(self.atoms)

    def __str__(self):
        if self.smiles is not None:
            return f"Adsorbate: ({self.atoms.get_chemical_formula()}, {self.smiles})"
        else:
            return f"Adsorbate: ({self.atoms.get_chemical_formula()})"

    def __repr__(self):
        return self.__str__()

import os
import pickle
import warnings

import ase
import numpy as np

from ocdata.configs.paths import ADSORBATES_PKL_PATH


class Adsorbate:
    """
    Initializes an adsorbate object in one of 4 ways:
    - Directly pass in an ase.Atoms object. For this, you should also
        provide a binding index (see Args).
    - Pass in index of adsorbate to select from adsorbate database.
    - Pass in the SMILES string of the adsorbate to select from the database.
    - Randomly sample an adsorbate from adsorbate database.

    Arguments
    ---------
    adsorbate_atoms: ase.Atoms
        Adsorbate structure.
    adsorbate_id_from_db: int
        Index of adsorbate to select.
    adsorbate_smiles_from_db: str
        A SMILES string of the desired adsorbate.
    adsorbate_db_path: str
        Path to adsorbate database.
    adsorbate_binding_indices: list
        The index/indices of the adsorbate atoms which are expected to bind.
    """

    def __init__(
        self,
        adsorbate_atoms: ase.Atoms = None,
        adsorbate_id_from_db: int = None,
        adsorbate_smiles_from_db: str = None,
        adsorbate_db_path: str = ADSORBATES_PKL_PATH,
        adsorbate_binding_indices: list = None,
    ):
        self.adsorbate_id_from_db = adsorbate_id_from_db
        self.adsorbate_db_path = adsorbate_db_path

        if adsorbate_atoms is None and adsorbate_binding_indices is not None:
            warnings.warn(
                "adsorbates from the database have predefined binding indexes, those will be used instead."
            )

        if adsorbate_atoms is not None:
            self.atoms = adsorbate_atoms.copy()
            self.smiles = None
            if adsorbate_binding_indices is None:
                random_idx = np.random.randint(len(adsorbate_atoms))
                self.binding_indices = [random_idx]
                warnings.warn(
                    "\nNo binding index was provided, so one was chosen at random.\n"
                    "If you plan to use heuristic placement, this may cause unexpected behavior.\n"
                    f"The binding atom index is {random_idx} "
                    f"and the chemical symbol is {adsorbate_atoms.get_chemical_symbols()[random_idx]}"
                )
            else:
                self.binding_indices = adsorbate_binding_indices
        elif adsorbate_id_from_db is not None:
            adsorbate_db = pickle.load(open(adsorbate_db_path, "rb"))
            self.atoms, self.smiles, self.binding_indices = adsorbate_db[
                adsorbate_id_from_db
            ]
        elif adsorbate_smiles_from_db is not None:
            adsorbate_db = pickle.load(open(adsorbate_db_path, "rb"))
            adsorbate_obj_tuple = [
                (idx, adsorbate_info)
                for idx, adsorbate_info in adsorbate_db.items()
                if adsorbate_info[1] == adsorbate_smiles_from_db
            ]
            if len(adsorbate_obj_tuple) < 1:
                warnings.warn(
                    "An adsorbate with that SMILES string was not found. Choosing one at random instead."
                )
                self._get_adsorbate_from_random(adsorbate_db)
            else:
                self.atoms, self.smiles, self.binding_indices = adsorbate_obj_tuple[0][
                    1
                ]
                self.adsorbate_id_from_db = adsorbate_obj_tuple[0][0]
        else:
            adsorbate_db = pickle.load(open(adsorbate_db_path, "rb"))
            self._get_adsorbate_from_random(adsorbate_db)

    def __len__(self):
        return len(self.atoms)

    def __str__(self):
        if self.smiles is not None:
            return f"Adsorbate: ({self.atoms.get_chemical_formula()}, {self.smiles})"
        else:
            return f"Adsorbate: ({self.atoms.get_chemical_formula()})"

    def __repr__(self):
        return self.__str__()

    def _get_adsorbate_from_random(self, adsorbate_db):
        self.adsorbate_id_from_db = np.random.randint(len(adsorbate_db))
        self.atoms, self.smiles, self.binding_indices = adsorbate_db[
            self.adsorbate_id_from_db
        ]


def randomly_rotate_adsorbate(
    adsorbate_atoms: ase.Atoms, mode: str = "random", binding_idx: int = None
):
    atoms = adsorbate_atoms.copy()
    # To sample uniformly random 3D rotations, we first sample a uniformly
    # random rotation about the z-axis. Then, rotate the unmoved north pole to a
    # random position. This also makes it easier to implement the "heuristic"
    # mode the second step can be changed to sample rotations only within a
    # certain cone around the north pole.

    if mode == "random":
        # Rotate uniformly about center of mass along all three directions.
        zrot = np.random.uniform(0, 360)
        atoms.rotate(zrot, "z", center="COM")
        z = np.random.uniform(-1.0, 1.0)
        phi = np.random.uniform(0, 2 * np.pi)
        rotvec = np.array(
            [np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z]
        )
        atoms.rotate(a=(0, 0, 1), v=rotvec, center="COM")
    elif mode in ["heuristic", "random_site_heuristic_placement"]:
        assert binding_idx is not None
        # Rotate uniformly about binding atom along the z-axis, but only
        # slight wobbles around x and y, to avoid crashing into the surface.
        zrot = np.random.uniform(0, 360)
        atoms.rotate(zrot, "z", center=atoms.positions[binding_idx])
        # PI / 9 was arbitrarily chosen as the cone angle.
        z = np.random.uniform(np.cos(np.pi / 9), 1.0)
        phi = np.random.uniform(0, 2 * np.pi)
        rotvec = np.array(
            [np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z]
        )
        atoms.rotate(a=(0, 0, 1), v=rotvec, center=atoms.positions[binding_idx])
    else:
        raise NotImplementedError

    return atoms, [zrot, rotvec]

import os
import pickle

import ase
import numpy as np
import pymatgen
from pymatgen.core.surface import (
    SlabGenerator,
    get_symmetrically_distinct_miller_indices,
)
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ocdata.configs.constants import COVALENT_MATERIALS_MPIDS, MAX_MILLER
from ocdata.configs.paths import BULK_PKL_PATH, PRECOMPUTED_SLABS_DIR_PATH


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
    mp_id: str
        Materials Project ID of bulk to select if not doing a random sample.
        [CURRENTLY NOT SUPPORTED.]
    """

    def __init__(
        self,
        bulk_atoms: ase.Atoms = None,
        bulk_id_from_db: int = None,
        # mp_id = None,
        bulk_db_path: str = BULK_PKL_PATH,
        precomputed_slabs_path: str = PRECOMPUTED_SLABS_DIR_PATH,
    ):
        self.bulk_id_from_db = bulk_id_from_db
        self.bulk_db_path = bulk_db_path
        self.precomputed_slabs_path = precomputed_slabs_path

        if bulk_atoms is not None:
            self.atoms = bulk_atoms.copy()
            self.mpid = None
        elif bulk_id_from_db is not None:
            bulk_db = pickle.load(open(bulk_db_path, "rb"))
            (
                self.atoms,
                self.mpid,
                _,
                _,
            ) = bulk_db[bulk_id_from_db]
        else:
            bulk_db = pickle.load(open(bulk_db_path, "rb"))
            bulk_id_from_db = np.random.randint(len(bulk_db))
            (
                self.atoms,
                self.mpid,
                _,
                _,
            ) = bulk_db[bulk_id_from_db]
            self.bulk_id_from_db = bulk_id_from_db

        # Comment(@abhshkdz): Do we need this?
        self.n_elems = len(set(self.atoms.symbols))

    def set_mpid(self, mpid: str):
        self.mpid = mpid

    def set_bulk_id_from_db(self, bulk_id_from_db: int):
        self.bulk_id_from_db = bulk_id_from_db

    def get_slabs(self):
        """
        Returns a list of possible slabs for this bulk instance.
        This can be later used to iterate through all slabs,
        or select one at random, to make a Surface object.
        """
        if self.precomputed_slabs_path is not None and self.bulk_id_from_db is not None:
            slabs = self.get_precomputed_slabs()
        else:
            slabs = self.compute_slabs()

        return slabs

    def get_precomputed_slabs(self):
        """
        Loads relevant pickle of precomputed slabs, and returns
        a list of tuples containing: (atoms, miller, shift, top).
        """
        assert self.bulk_id_from_db is not None

        bulk_slabs_path = os.path.join(
            self.precomputed_slabs_path, f"{self.bulk_id_from_db}.pkl"
        )
        assert os.path.exists(bulk_slabs_path)

        with open(bulk_slabs_path, "rb") as f:
            slabs_info = pickle.load(f)

        return slabs_info

    def compute_slabs(self, max_miller=MAX_MILLER):
        """
        Enumerates all the symmetrically distinct slabs of a bulk structure.
        It will not enumerate slabs with Miller indices above the
        `max_miller` argument. Note that we also look at the bottoms of slabs
        if they are distinct from the top. If they are distinct, we flip the
        surface so the bottom is pointing upwards.

        Args:
            max_miller  An integer indicating the maximum Miller index of the slabs
                        you are willing to enumerate. Increasing this argument will
                        increase the number of slabs, but the slabs will
                        generally become larger.
        Returns:
            all_slabs_info  A list of 4-tuples containing:  `pymatgen.Structure`
                            objects for slabs we have enumerated, the Miller
                            indices, floats for the shifts, and Booleans for "top".
        """
        bulk_struct = standardize_bulk(self.atoms)

        all_slabs_info = []
        for millers in get_symmetrically_distinct_miller_indices(
            bulk_struct, MAX_MILLER
        ):
            slab_gen = SlabGenerator(
                initial_structure=bulk_struct,
                miller_index=millers,
                min_slab_size=7.0,
                min_vacuum_size=20.0,
                lll_reduce=False,
                center_slab=True,
                primitive=True,
                max_normal_search=1,
            )
            slabs = slab_gen.get_slabs(
                tol=0.3, bonds=None, max_broken_bonds=0, symmetrize=False
            )

            # Additional filtering for the 2D materials' slabs
            if self.mpid is not None and self.mpid in COVALENT_MATERIALS_MPIDS:
                slabs = [slab for slab in slabs if is_2D_slab_reasonable(slab) is True]

            # If the bottom of the slabs are different than the tops, then we
            # want to consider them too.
            if len(slabs) != 0:
                flipped_slabs_info = [
                    (flip_struct(slab), millers, slab.shift, False)
                    for slab in slabs
                    if is_structure_invertible(slab) is False
                ]

                # Concatenate all the results together
                slabs_info = [(slab, millers, slab.shift, True) for slab in slabs]
                all_slabs_info.extend(slabs_info + flipped_slabs_info)

        return all_slabs_info


def is_2D_slab_reasonable(struct: pymatgen.Structure):
    """
    There are 400+ 2D bulk materials whose slabs generated by pymatgen require
    additional filtering: some slabs are cleaved where one or more surface atoms
    have no bonds with other atoms on the slab.

    Arguments
    ---------
    struct: pymatgen.Structure
        `pymatgen.Structure` object of a slab

    Returns
    -------
    A boolean indicating whether or not the slab is reasonable, where
    reasonable is defined as having at least one neighboring atom within 3A.
    """
    for site in struct:
        if len(struct.get_neighbors(site, 3)) == 0:
            return False
    return True


def standardize_bulk(atoms: ase.Atoms):
    """
    There are many ways to define a bulk unit cell. If you change the unit
    cell itself but also change the locations of the atoms within the unit
    cell, you can effectively get the same bulk structure. To address this,
    there is a standardization method used to reduce the degrees of freedom
    such that each unit cell only has one "true" configuration. This
    function will align a unit cell you give it to fit within this
    standardization.

    Arguments
    ---------
    atoms: ase.Atoms
        `ase.Atoms` object of the bulk you want to standardize

    Returns
    -------
    standardized_struct: pymatgen.Structure
        `pymatgen.Structure` of the standardized bulk
    """
    struct = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(struct, symprec=0.1)
    standardized_struct = sga.get_conventional_standard_structure()
    return standardized_struct


def flip_struct(struct: pymatgen.Structure):
    """
    Flips an atoms object upside down. Normally used to flip slabs.

    Arguments
    ---------
    struct: pymatgen.Structure
        `pymatgen.Structure` object of the surface you want to flip

    Returns
    -------
    flipped_struct: pymatgen.Structure
        `pymatgen.Structure` object of the flipped surface.
    """
    atoms = AseAtomsAdaptor.get_atoms(struct)

    # This is black magic wizardry to me. Good look figuring it out.
    atoms.wrap()
    atoms.rotate(180, "x", rotate_cell=True, center="COM")
    if atoms.cell[2][2] < 0.0:
        atoms.cell[2] = -atoms.cell[2]
    if np.cross(atoms.cell[0], atoms.cell[1])[2] < 0.0:
        atoms.cell[1] = -atoms.cell[1]
    atoms.center()
    atoms.wrap()

    return AseAtomsAdaptor.get_structure(atoms)


def is_structure_invertible(struct: pymatgen.Structure):
    """
    This function figures out whether or not an `pymatgen.Structure` object has
    symmetricity. In this function, the affine matrix is a rotation matrix that
    is multiplied with the XYZ positions of the crystal. If the z,z component
    of that is negative, it means symmetry operation exist, it could be a
    mirror operation, or one that involves multiple rotations/etc. Regardless,
    it means that the top becomes the bottom and vice-versa, and the structure
    is the symmetric. i.e. structure_XYZ = structure_XYZ*M.

    In short:  If this function returns `False`, then the input structure can
    be flipped in the z-direction to create a new structure.

    Arguments
    ---------
    struct: pymatgen.Structure

    Returns
    -------
    A boolean indicating whether or not your `ase.Atoms` object is
    symmetric in z-direction (i.e. symmetric with respect to x-y plane).
    """
    # If any of the operations involve a transformation in the z-direction,
    # then the structure is invertible.
    sga = SpacegroupAnalyzer(struct, symprec=0.1)
    for operation in sga.get_symmetry_operations():
        xform_matrix = operation.affine_matrix
        z_xform = xform_matrix[2, 2]
        if z_xform == -1:
            return True
    return False

'''
This submodule contains the scripts that the we used to sample the adsorption
structures.

Note that some of these scripts were taken from
[GASpy](https://github.com/ulissigroup/GASpy) with permission of author.
'''

__authors__ = ['Kevin Tran', 'Aini Palizhati', 'Siddharth Goyal', 'Zachary Ulissi']
__email__ = ['ktran@andrew.cmu.edu']

import math
from collections import defaultdict
import random
import pickle
import numpy as np
import catkit
import ase
import ase.db
from ase import neighborlist
from ase.constraints import FixAtoms
from ase.neighborlist import natural_cutoffs
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN
from base_atoms.pkls import BULK_PKL, ADSORBATE_PKL
import sys
import time

ELEMENTS = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
            9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
            23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29:
            'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br',
            36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42:
            'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd',
            49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55:
            'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm',
            62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68:
            'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W',
            75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81:
            'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr',
            88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94:
            'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
            101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg',
            107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn',
            113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'}

# Covalent radius of elements (unit is pm, 1pm=0.01 angstrom)
# Value are taken from https://github.com/lmmentel/mendeleev
covalent_radius = {'H': 32.0, 'He': 46.0, 'O': 63.0, 'F': 64.0, 'Ne': 67.0, 'N': 71.0,
                   'C': 75.0, 'B': 85.0, 'Ar': 96.0, 'Cl': 99.0, 'Be': 102.0, 'S': 103.0,
                   'Ni': 110.0, 'P': 111.0, 'Co': 111.0, 'Cu': 112.0, 'Br': 114.0,
                   'Si': 116.0, 'Fe': 116.0, 'Se': 116.0, 'Kr': 117.0, 'Zn': 118.0,
                   'Mn': 119.0, 'Pd': 120.0, 'Ge': 121.0, 'As': 121.0, 'Rg': 121.0,
                   'Cr': 122.0, 'Ir': 122.0, 'Cn': 122.0, 'Pt': 123.0, 'Ga': 124.0,
                   'Au': 124.0, 'Ru': 125.0, 'Rh': 125.0, 'Al': 126.0, 'Tc': 128.0,
                   'Ag': 128.0, 'Ds': 128.0, 'Os': 129.0, 'Mt': 129.0, 'Xe': 131.0, 'Re': 131.0,
                   'Li': 133.0, 'I': 133.0, 'Hg': 133.0, 'V': 134.0, 'Hs': 134.0, 'Ti': 136.0,
                   'Cd': 136.0, 'Te': 136.0, 'Nh': 136.0, 'W': 137.0, 'Mo': 138.0, 'Mg': 139.0,
                   'Sn': 140.0, 'Sb': 140.0, 'Bh': 141.0, 'In': 142.0, 'Rn': 142.0, 'Sg': 143.0,
                   'Fl': 143.0, 'Tl': 144.0, 'Pb': 144.0, 'Po': 145.0, 'Ta': 146.0, 'Nb': 147.0,
                   'At': 147.0, 'Sc': 148.0, 'Db': 149.0, 'Bi': 151.0, 'Hf': 152.0, 'Zr': 154.0,
                   'Na': 155.0, 'Rf': 157.0, 'Og': 157.0, 'Lr': 161.0, 'Lu': 162.0, 'Mc': 162.0,
                   'Y': 163.0, 'Ce': 163.0, 'Tm': 164.0, 'Er': 165.0, 'Es': 165.0, 'Ts': 165.0,
                   'Ho': 166.0, 'Am': 166.0, 'Cm': 166.0, 'Dy': 167.0, 'Fm': 167.0, 'Eu': 168.0,
                   'Tb': 168.0, 'Bk': 168.0, 'Cf': 168.0, 'Gd': 169.0, 'Pa': 169.0, 'Yb': 170.0,
                   'U': 170.0, 'Ca': 171.0, 'Np': 171.0, 'Sm': 172.0, 'Pu': 172.0, 'Pm': 173.0,
                   'Md': 173.0, 'Nd': 174.0, 'Th': 175.0, 'Lv': 175.0, 'Pr': 176.0, 'No': 176.0,
                   'La': 180.0, 'Sr': 185.0, 'Ac': 186.0, 'K': 196.0, 'Ba': 196.0, 'Ra': 201.0,
                   'Rb': 210.0, 'Fr': 223.0, 'Cs': 232.0}

# We will enumerate surfaces with Miller indices <= MAX_MILLER
MAX_MILLER = 2

# We will create surfaces that are at least MIN_XY Angstroms wide. GASpy uses
# 4.5, but our larger adsorbates here can be up to 3.6 Angstroms long. So 4.5 +
# 3.6 ~= 8 Angstroms
MIN_XY = 8.



def enumerate_surfaces_for_saving(bulk_atoms, max_miller=MAX_MILLER):
    '''
    Enumerate all the symmetrically distinct surfaces of a bulk structure. It
    will not enumerate surfaces with Miller indices above the `max_miller`
    argument. Note that we also look at the bottoms of surfaces if they are
    distinct from the top. If they are distinct, we flip the surface so the bottom
    is pointing upwards.

    Args:
        bulk_atoms  `ase.Atoms` object of the bulk you want to enumerate
                    surfaces from.
        max_miller  An integer indicating the maximum Miller index of the surfaces
                    you are willing to enumerate. Increasing this argument will
                    increase the number of surfaces, but the surfaces will
                    generally become larger.
    Returns:
        all_slabs_info  A list of 4-tuples containing:  `pymatgen.Structure`
                        objects for surfaces we have enumerated, the Miller
                        indices, floats for the shifts, and Booleans for "top".
    '''
    bulk_struct = standardize_bulk(bulk_atoms)

    all_slabs_info = []
    for millers in get_symmetrically_distinct_miller_indices(bulk_struct, MAX_MILLER):
        slab_gen = SlabGenerator(initial_structure=bulk_struct,
                                 miller_index=millers,
                                 min_slab_size=7.,
                                 min_vacuum_size=20.,
                                 lll_reduce=False,
                                 center_slab=True,
                                 primitive=True,
                                 max_normal_search=1)
        slabs = slab_gen.get_slabs(tol=0.3,
                                   bonds=None,
                                   max_broken_bonds=0,
                                   symmetrize=False)

        # If the bottoms of the slabs are different than the tops, then we want
        # to consider them, too
        flipped_slabs_info = [(flip_struct(slab), millers, slab.shift, False)
                              for slab in slabs if is_structure_invertible(slab) is False]

        # Concatenate all the results together
        slabs_info = [(slab, millers, slab.shift, True) for slab in slabs]
        all_slabs_info.extend(slabs_info + flipped_slabs_info)
    return all_slabs_info


def standardize_bulk(atoms):
    '''
    There are many ways to define a bulk unit cell. If you change the unit cell
    itself but also change the locations of the atoms within the unit cell, you
    can get effectively the same bulk structure. To address this, there is a
    standardization method used to reduce the degrees of freedom such that each
    unit cell only has one "true" configuration. This function will align a
    unit cell you give it to fit within this standardization.

    Arg:
        atoms   `ase.Atoms` object of the bulk you want to standardize
    Returns:
        standardized_struct     `pymatgen.Structure` of the standardized bulk
    '''
    struct = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(struct, symprec=0.1)
    standardized_struct = sga.get_conventional_standard_structure()
    return standardized_struct


def is_structure_invertible(structure):
    '''
    This function figures out whether or not an `pymatgen.Structure` object has
    symmetricity. In this function, the affine matrix is a rotation matrix that
    is multiplied with the XYZ positions of the crystal. If the z,z component
    of that is negative, it means symmetry operation exist, it could be a
    mirror operation, or one that involves multiple rotations/etc. Regardless,
    it means that the top becomes the bottom and vice-versa, and the structure
    is the symmetric. i.e. structure_XYZ = structure_XYZ*M.

    In short:  If this function returns `False`, then the input structure can
    be flipped in the z-direction to create a new structure.

    Arg:
        structure   A `pymatgen.Structure` object.
    Returns
        A boolean indicating whether or not your `ase.Atoms` object is
        symmetric in z-direction (i.e. symmetric with respect to x-y plane).
    '''
    # If any of the operations involve a transformation in the z-direction,
    # then the structure is invertible.
    sga = SpacegroupAnalyzer(structure, symprec=0.1)
    for operation in sga.get_symmetry_operations():
        xform_matrix = operation.affine_matrix
        z_xform = xform_matrix[2, 2]
        if z_xform == -1:
            return True
    return False


def flip_struct(struct):
    '''
    Flips an atoms object upside down. Normally used to flip surfaces.

    Arg:
        atoms   `pymatgen.Structure` object
    Returns:
        flipped_struct  The same `ase.Atoms` object that was fed as an
                        argument, but flipped upside down.
    '''
    atoms = AseAtomsAdaptor.get_atoms(struct)

    # This is black magic wizardry to me. Good look figuring it out.
    atoms.wrap()
    atoms.rotate(180, 'x', rotate_cell=True, center='COM')
    if atoms.cell[2][2] < 0.:
        atoms.cell[2] = -atoms.cell[2]
    if np.cross(atoms.cell[0], atoms.cell[1])[2] < 0.0:
        atoms.cell[1] = -atoms.cell[1]
    atoms.wrap()

    flipped_struct = AseAtomsAdaptor.get_structure(atoms)
    return flipped_struct

    
def precompute_enumerate_surface(bulk_database, bulk_index, opfile):

    with open(bulk_database, 'rb') as f:
        inv_index = pickle.load(f)
    flatten = inv_index[1] + inv_index[2] + inv_index[3]
    assert bulk_index < len(flatten)

    bulk, mpid = flatten[bulk_index] 

    print(bulk, mpid)
    surfaces_info = enumerate_surfaces_for_saving(bulk)

    with open(opfile, 'wb') as g:
        pickle.dump(surfaces_info, g)

s = time.time()
precompute_enumerate_surface(BULK_PKL, int(sys.argv[1]), sys.argv[2])
e = time.time()
print(sys.argv[1], "Done in", e - s )

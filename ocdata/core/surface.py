import math
import os
import pickle
from collections import defaultdict

import numpy as np
from ase import neighborlist
from ase.constraints import FixAtoms
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ocdata.configs.constants import MIN_XY
from ocdata.core import Bulk


class Surface:
    """
    Initializes a surface object, i.e. a particular slab tiled along xyz, in
    one of 2 ways:
    - Pass in a Bulk object and a slab 4-tuple containing
    (atoms, miller, shift, top).
    - Pass in a Bulk object and randomly sample a surface.

    Arguments
    ---------
    bulk: Bulk
        Corresponding bulk object.
    slab: tuple
        4-tuple containing (atoms, miller, shift, top).
    """

    def __init__(
        self,
        bulk: Bulk,
        slab: tuple = None,
    ):
        self.bulk = bulk
        self.slab = slab

        assert bulk is not None

        if slab is not None:
            # Comment(@abhshkdz): do all of these need to be class attributes?
            self.unit_surface_struct, self.millers, self.shift, self.top = slab
        else:
            slabs = bulk.get_slabs()
            slab_idx = np.random.randint(len(slabs))
            self.unit_surface_struct, self.millers, self.shift, self.top = slabs[
                slab_idx
            ]

        self.unit_surface_atoms = AseAtomsAdaptor.get_atoms(self.unit_surface_struct)
        self.atoms = tile_atoms(self.unit_surface_atoms)

        assert (
            Composition(self.atoms.get_chemical_formula()).reduced_formula
            == Composition(bulk.atoms.get_chemical_formula()).reduced_formula
        ), "Mismatched bulk and surface"

        self.tag_surface_atoms()
        self.set_fixed_atom_constraints()

    def tag_surface_atoms(self):
        """
        Sets the tags of an `ase.Atoms` object. Any atom that we consider a "bulk"
        atom will have a tag of 0, and any atom that we consider a "surface" atom
        will have a tag of 1. We use a combination of Voronoi neighbor algorithms
        (adapted from from `pymatgen.core.surface.Slab.get_surface_sites`; see
        https://pymatgen.org/pymatgen.core.surface.html) and a distance cutoff.

        Arg:
            bulk_atoms      `ase.Atoms` format of the respective bulk structure
            surface_atoms   The surface where you are trying to find surface sites in
                            `ase.Atoms` format
        """
        voronoi_tags = find_surface_atoms_with_voronoi(self.bulk.atoms, self.atoms)
        height_tags = find_surface_atoms_by_height(self.atoms)
        # If either of the methods consider an atom a "surface atom", then tag it as such.
        tags = [max(v_tag, h_tag) for v_tag, h_tag in zip(voronoi_tags, height_tags)]
        self.atoms.set_tags(tags)

    def set_fixed_atom_constraints(self):
        """
        This function fixes sub-surface atoms of a surface. Also works on systems
        that have surface + adsorbate(s), as long as the bulk atoms are tagged with
        `0`, surface atoms are tagged with `1`, and the adsorbate atoms are tagged
        with `2` or above.

        This function is used for both surface atoms and the combined surface+adsorbate

        Inputs:
            atoms           `ase.Atoms` class of the surface system. The tags of
                            these atoms must be set such that any bulk/surface
                            atoms are tagged with `0` or `1`, resectively, and any
                            adsorbate atom is tagged with a 2 or above.
        Returns:
            atoms           A deep copy of the `atoms` argument, but where the appropriate
                            atoms are constrained.
        """
        # We'll be making a `mask` list to feed to the `FixAtoms` class. This
        # list should contain a `True` if we want an atom to be constrained, and
        # `False` otherwise.
        mask = [True if atom.tag == 0 else False for atom in self.atoms]
        self.atoms.constraints += [FixAtoms(mask=mask)]

    def __len__(self):
        return len(self.atoms)

    def __str__(self):
        return f"Surface: {self.atoms.get_chemical_formula()}"

    def __repr__(self):
        return self.__str__()


def tile_atoms(atoms, min_xy=MIN_XY):
    """
    This function will repeat an atoms structure in the x and y direction until
    the x and y dimensions are at least as wide as the MIN_XY constant.

    Arguments
    ---------
    atoms: ase.Atoms
        The structure to tile.

    Returns
    -------
    atoms_tiled: ase.Atoms
        The tiled structure.
    """
    x_length = np.linalg.norm(atoms.cell[0])
    y_length = np.linalg.norm(atoms.cell[1])
    nx = int(math.ceil(min_xy / x_length))
    ny = int(math.ceil(min_xy / y_length))
    n_xyz = (nx, ny, 1)
    atoms_tiled = atoms.repeat(n_xyz)
    return atoms_tiled


def find_surface_atoms_by_height(surface_atoms):
    """
    As discussed in the docstring for `find_surface_atoms_with_voronoi`,
    sometimes we might accidentally tag a surface atom as a bulk atom if there
    are multiple coordination environments for that atom type within the bulk.
    One heuristic that we use to address this is to simply figure out if an
    atom is close to the surface. This function will figure that out.

    Specifically:  We consider an atom a surface atom if it is within 2
    Angstroms of the heighest atom in the z-direction (or more accurately, the
    direction of the 3rd unit cell vector).

    Arguments
    ---------
    surface_atoms: ase.Atoms

    Returns
    -------
    tags: list
        A list that contains the indices of the surface atoms.
    """
    unit_cell_height = np.linalg.norm(surface_atoms.cell[2])
    scaled_positions = surface_atoms.get_scaled_positions()
    scaled_max_height = max(scaled_position[2] for scaled_position in scaled_positions)
    scaled_threshold = scaled_max_height - 2.0 / unit_cell_height

    tags = [
        0 if scaled_position[2] < scaled_threshold else 1
        for scaled_position in scaled_positions
    ]
    return tags


def find_surface_atoms_with_voronoi(bulk_atoms, surface_atoms):
    """
    Labels atoms as surface or bulk atoms according to their coordination
    relative to their bulk structure. If an atom's coordination is less than it
    normally is in a bulk, then we consider it a surface atom. We calculate the
    coordination using pymatgen's Voronoi algorithms.

    Note that if a single element has different sites within a bulk and these
    sites have different coordinations, then we consider slab atoms
    "under-coordinated" only if they are less coordinated than the most under
    undercoordinated bulk atom. For example:  Say we have a bulk with two Cu
    sites. One site has a coordination of 12 and another a coordination of 9.
    If a slab atom has a coordination of 10, we will consider it a bulk atom.

    Arguments
    ---------
    bulk_atoms: ase.Atoms
        The bulk structure that the surface was cut from.
    surface_atoms: ase.Atoms
        The surface structure.

    Returns
    -------
    tags: list
        A list of 0s and 1s whose indices align with the atoms in
        `surface_atoms`. 0s indicate a bulk atom and 1 indicates a surface atom.
    """
    # Initializations
    surface_struct = AseAtomsAdaptor.get_structure(surface_atoms)
    center_of_mass = calculate_center_of_mass(surface_struct)
    bulk_cn_dict = calculate_coordination_of_bulk_atoms(bulk_atoms)
    voronoi_nn = VoronoiNN(tol=0.1)  # 0.1 chosen for better detection

    tags = []
    for idx, site in enumerate(surface_struct):

        # Tag as surface atom only if it's above the center of mass
        if site.frac_coords[2] > center_of_mass[2]:
            try:

                # Tag as surface if atom is under-coordinated
                cn = voronoi_nn.get_cn(surface_struct, idx, use_weights=True)
                cn = round(cn, 5)
                if cn < min(bulk_cn_dict[site.species_string]):
                    tags.append(1)
                else:
                    tags.append(0)

            # Tag as surface if we get a pathological error
            except RuntimeError:
                tags.append(1)

        # Tag as bulk otherwise
        else:
            tags.append(0)
    return tags


def calculate_center_of_mass(struct):
    """
    Determine the surface atoms indices from here
    """
    weights = [site.species.weight for site in struct]
    center_of_mass = np.average(struct.frac_coords, weights=weights, axis=0)
    return center_of_mass


def calculate_coordination_of_bulk_atoms(bulk_atoms):
    """
    Finds all unique atoms in a bulk structure and then determines their
    coordination number. Then parses these coordination numbers into a
    dictionary whose keys are the elements of the atoms and whose values are
    their possible coordination numbers.
    For example: `bulk_cns = {'Pt': {3., 12.}, 'Pd': {12.}}`

    Arguments
    ---------
    bulk_atoms: ase.Atoms
        The bulk structure.

    Returns
    -------
    bulk_cn_dict: dict
        A dictionary whose keys are the elements of the atoms and whose values
        are their possible coordination numbers.
    """
    voronoi_nn = VoronoiNN(tol=0.1)  # 0.1 chosen for better detection

    # Object type conversion so we can use Voronoi
    bulk_struct = AseAtomsAdaptor.get_structure(bulk_atoms)
    sga = SpacegroupAnalyzer(bulk_struct)
    sym_struct = sga.get_symmetrized_structure()

    # We'll only loop over the symmetrically distinct sites for speed's sake
    bulk_cn_dict = defaultdict(set)
    for idx in sym_struct.equivalent_indices:
        site = sym_struct[idx[0]]
        cn = voronoi_nn.get_cn(sym_struct, idx[0], use_weights=True)
        cn = round(cn, 5)
        bulk_cn_dict[site.species_string].add(cn)
    return bulk_cn_dict

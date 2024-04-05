import os
import subprocess
import tempfile
from pathlib import Path

import ase.io
import numpy as np


def get_molecules_per_volume(density, molar_mass):
    """
    Given a molecular density (g/ml) and molar mass (g/mol), return the number
    of molecules necessary per cubic angstrom to preserve the density.

    Arguments:
        density: float
            Molecular density of molecule, in g/mL
        molar_mass: float
            Molar mass of molecule, in g/mol

    Outputs:
        # molecules per unit volume
    """
    return (6.022e23) * density / (1e24 * molar_mass)


def _create_structure_block(structure_path, num_molecules, box_size):
    """
    For each unique molecular structure to include in the packmol input, this
    block must be created.

    Arguments:
        structure_path: str
            Path to the molecular structure to be included in packmol. PDB
            format.
        num_molecules: int
            # of molecules to include in the specified box size
        box_size: str
            Box to pack molecules in. Specified as:
            "{x_min} {y_min} {z_min} {x_max} {y_max} {z_max}" where each
            variable is a float corresponding to min/max of the box, in
            angstroms.
    """
    block = f"structure {structure_path}\n  number {num_molecules}\n  inside box {box_size}\nend structure\n\n"
    return block


def create_packmol_inputs(list_of_structures, outdir):
    template_inp = Path(
        "/private/home/mshuaibi/projects/ocpdata/ocdata/solvents/packmol_template.inp"
    )
    input_lines = template_inp.read_text()
    input_lines += f"output {os.path.join(outdir, 'packmol_outputs.pdb')}\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, structure_info in enumerate(list_of_structures):
            atoms, num_molecules, box_size = structure_info
            atoms_path = os.path.join(tmpdir, f"{i}.pdb")
            ase.io.write(atoms_path, atoms)

            _input_structure_lines = _create_structure_block(
                atoms_path, num_molecules, box_size
            )
            input_lines += _input_structure_lines

        new_path = Path(os.path.join(tmpdir, "packmol_inputs.inp"))
        new_path.write_text(input_lines)

        out, err = run_packmol(new_path)
    return out, err


def run_packmol(path):
    ps = subprocess.Popen(
        f"/private/home/mshuaibi/projects/packmol/packmol-20.14.3/packmol < {path}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = ps.communicate()
    return out, err


def get_box_dimensions(slab):
    """
    Compute the desired box dimensions, given an input slab to be solvated.
    """
    top_atoms = slab.atoms.positions[slab.atoms.get_tags() == 1]
    max_height = np.argmax(top_atoms[:, 2])
    z_min = top_atoms[max_height][2] + 2
    z_max = slab.atoms.get_cell()[2][2]
    x_max = slab.atoms.get_cell()[0][0]
    y_max = slab.atoms.get_cell()[1][1]

    return x_max, y_max, z_min, z_max

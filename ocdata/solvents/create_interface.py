import os
import ase
from ase.cell import Cell
from ase import Atoms
from ase.io import read
from geometry import BoxGeometry, PlaneBoundTriclinicGeometry
import tempfile
from shutil import copyfile
from clint.textui import progress
from werkzeug.utils import secure_filename
import subprocess
from ocdata.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab
from ase.build import add_adsorbate


"function adapted from https://github.com/henriasv/molecular-builder/tree/master"


def pack_water(atoms=None, nummol=None, volume=None, density=1.0,
               geometry=None, side='in', pbc=0.0, tolerance=2.0):
    """Pack water molecules into voids at a given volume defined by a geometry.
    The packing is performed by packmol.

    :param atoms: ase Atoms object that specifies particles that water is to be packed around. The packed water molecules will be added to this atoms object.
    :type atoms: Atoms object
    :param nummol: Number of water molecules
    :type nummol: int
    :param volume: Void volume in :math:`Å^3` to be filled with water. Can only be used if `nummol=None`, since `pack_water` will compute the number of atoms based on the volume and density of water.
    :type volume: float
    :param density: Water density. Used to compute the number of water molecules to be packed if `volume` is provided.
    :type density: float
    :param geometry: Geometry object specifying where to pack water
    :type geometry: Geometry object
    :param side: Pack water inside/outside of geometry
    :type side: str
    :param pbc: Inner margin to add to the simulation box to avoid overlapping atoms over periodic boundary conditions. This is necessary because packmol doesn't support periodic boundary conditions.
    :type pbc: float or array_like
    :param tolerance: Minimum separation distance between molecules.
    :type tolerance: float

    :returns: Coordinates of the packed water
    """
    if (volume is None and nummol is None):
        raise ValueError("Either volume or the number of molecules needed")
    elif (volume is not None) and (nummol is not None):
        raise ValueError("Either volume or the number of molecules needed")

    if volume is not None:
        V_per_water = 29.9796/density
        nummol = int(volume/V_per_water)

    format_s, format_v = "pdb", "proteindatabank"
    side += "side"

    if atoms is None and geometry is None:
        raise ValueError("Either atoms or geometry has to be given")
    elif geometry is None:
        # The default water geometry is a box which capsules the solid
        if type(pbc) is list or type(pbc) is tuple:
            pbc = np.array(pbc)

        cell = atoms.cell
        if cell.orthorhombic:
            box_length = cell.lengths()
            geometry = BoxGeometry(center=box_length/2, length=box_length - pbc)
        else:
            geometry = PlaneBoundTriclinicGeometry(cell, pbc=pbc)
    else:
        if geometry.__class__.__name__ == "PlaneBoundTriclinicGeometry":
            cell = geometry.cell
        else:
            cell = np.diag(geometry.ur_corner - geometry.ll_corner)

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)

        if atoms is not None and len(atoms) > 0:
            # Write solid structure to pdb-file
            atoms.write(f"atoms.{format_s}", format=format_v)

        # Copy water.pdb to temporary directory
        this_dir, this_filename = os.path.split(__file__)
        water_data = this_dir + f"/data_files/water.{format_s}"
        copyfile(water_data, f"water.{format_s}")

        # Generate packmol input script
        with open("input.inp", "w") as f:
            f.write(f"tolerance {tolerance}\n")
            f.write(f"filetype {format_s}\n")
            f.write(f"output out.{format_s}\n")
            if atoms is not None and len(atoms) > 0:
                f.write(f"structure atoms.{format_s}\n")
                f.write("  number 1\n")
                f.write("  fixed 0 0 0 0 0 0\n")
                f.write("end structure\n\n")
            f.write(geometry.packmol_structure(nummol, side))

        # Run packmol input script
        try:
            ps = subprocess.Popen("/p/lustre2/nitgo/packages/packmol-20.14.3/packmol < input.inp", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = ps.communicate()
        except:
            raise OSError("packmol is not found. For installation instructions, \
                           see http://m3g.iqm.unicamp.br/packmol/download.shtml.")

        # Read packmol outfile
        os.chdir(cwd)
        try:
            water = ase.io.read(f"{tmp_dir}/out.{format_s}", format=format_v)
            water.set_pbc(True)
        except FileNotFoundError as e: 
            raise FileNotFoundError(f"Packmol output not found\nOutput from Packmol: {out}\nError from Packmol: {err}\n{e}")

    os.chdir(cwd)
    if atoms is None:
        water.set_cell(cell)
    else:
        # remove solid
        if len(atoms) > 0:
            del water[:len(atoms)]
        water.set_cell(cell)
        atoms += water

    return water



if __name__ == "__main__":

   bulk = Bulk()

   slab = Slab.from_bulk_get_random_slab(bulk, max_miller=2, min_ab=8)

   atoms  = slab.atoms

   cell = atoms.get_cell()

   symbol = atoms.get_chemical_formula()

   water_box_cell = read('./data_files/water.pdb')

   z = 8.0

   water_box_cell.set_cell([cell[0],cell[1],[0.,0.,z]])

   water_box_cell = pack_water(volume=water_box_cell.get_volume(),atoms=water_box_cell)

   a = atoms.positions[:, 2].argmax()

   height = 3.0

   max_z = atoms.positions[a, 2] + height

   water_box_cell.translate([0,0,max_z])

   interface = atoms + water_box_cell

   interface.center(vacuum=15, axis=2) 

   interface.wrap()

   interface.write('{0}-interface.POSCAR'.format(symbol))


    


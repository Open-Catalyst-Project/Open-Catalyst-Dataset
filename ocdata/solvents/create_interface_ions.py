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

""" adapted vergion of Nitsh and Mohaumed code"""

"function adapted from https://github.com/henriasv/molecular-builder/tree/master"


def pack_solvent_ions(atoms=None,solvent=None,cation=None,anion=None, nummol=None, volume=None, density=None,
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

    def get_num_ions(cations,anions):
        """ 
        Give the number of cations and anions using to be charge neutral
        """
        ions_charges = {
                "Ca": 2,
                "Cs": 1,
                "F": -1,
                "H": 1,
                "HCO3": -1,
                "Li": 1,
                "OH": -1,
                "SO4": -2
                        }
        cation_value = ions_charges[cations]
        anions_value = ions_charges[anions]
        if cation_value == abs(anions_value):
            number_cations = 1
            number_anions = 1
            return number_cations,number_anions
        elif cation_value > abs(anions_value):
            number_cations = 1
            number_anions = cation_value // abs(anions_value)
            return number_cations,number_anions
        elif cation_value < abs(anions_value):
            number_anions = 1
            number_cations = abs(anions_value) // cation_value
            return number_cations,number_anions
        
    if (volume is None and nummol is None):
        raise ValueError("Either volume or the number of molecules needed")
    elif (volume is not None) and (nummol is not None):
        raise ValueError("Either volume or the number of molecules needed")

    if volume is not None:

        MMsolv = sum(solvent.get_masses())
        numol_p_V = get_molecules_per_volume(density,MMsolv)
        nummol = volume*numol_p_V
        nummol = int(np.round(nummol))


    format_s, format_v = "pdb", "proteindatabank"
    side += "side"
    if cation is not None and anion is not None:
        num_cation, num_anion = get_num_ions(str(cation.symbols),str(anion.symbols))

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
        if solvent is not None and len(solvent) > 0:
            solvent.write(f"solvent.{format_s}", format=format_v)

        if cation is not None and len(cation)>0:
            cation.write(f"cation.{format_s}",format=format_v)
            anion.write(f"anion.{format_s}",format=format_v)
        #this_dir, this_filename = os.path.split(__file__)
        #water_data = this_dir + f"/data_files/hexane1.{format_s}"
        #copyfile(water_data, f"solvent.{format_s}")

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
            print(geometry.packmol_structure(nummol, side))
            if cation is not None and anion is not None:
                f.write(geometry.packmol_structure_ions(num_cation, num_anion,side))
                print(geometry.packmol_structure_ions(num_cation, num_anion,side))

        # Run packmol input script
        try:
            ps = subprocess.Popen("/home/mmarasch/packmol-20.14.4/packmol < input.inp", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        #remove solid
        if len(atoms) > 0:
            del water[:len(atoms)]
        water.set_cell(cell)
        atoms += water

    return water



if __name__ == "__main__":
    import numpy as np
# An exemple for hexane
    bulk = Bulk()

    slab = Slab.from_bulk_get_random_slab(bulk, max_miller=2, min_ab=8)
    adsorbate = Adsorbate()
    atoms  = AdsorbateSlabConfig(slab, adsorbate, mode="random", num_sites=1)
    atoms =atoms.atoms_list[0]
    cell = atoms.get_cell()

    symbol = atoms.get_chemical_formula()

    solv = read('./data_files/hexanes.traj',":")
    solvent = solv[0]
    symbol_solv = solvent.get_chemical_formula()
    sovent_box_cell = solvent
    z = 8
    adsorbate =adsorbate.atoms
    adsorbate.set_cell([cell[0],cell[1],[0.,0.,z]])
    anion = read('./data_files/ions/SO4.pdb')
    cation = read('./data_files/ions/Li.pdb')

    solvent_box_cell = pack_solvent_ions(volume=adsorbate.get_volume(),atoms=adsorbate,solvent=solvent,cation=cation,anion=anion,density=0.66)
    print(1.66054*np.sum(solvent_box_cell.get_masses())/solvent_box_cell.get_volume())
    a = atoms.positions[:, 2].argmax()
    height = 3.0

    max_z = atoms.positions[a, 2] + height

    solvent_box_cell.translate([0,0,max_z])

    interface = atoms + solvent_box_cell

    interface.center(vacuum=15, axis=2) 

    interface.wrap()
    print('{0}-interface-{1}.POSCAR'.format(symbol,symbol_solv))
    interface.write('{0}-interface-{1}.POSCAR'.format(symbol,symbol_solv))


    


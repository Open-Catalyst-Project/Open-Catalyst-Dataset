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
import pickle
""" adapted vergion of Nitsh and Mohaumed code"""

"function adapted from https://github.com/henriasv/molecular-builder/tree/master"


def pack_solvent(atoms=None,solvent=None,ions=None, nummol=None, numions=None, volume=None, density=None,
               geometry=None, side='in', pbc=0.0, tolerance=2.0):
    """  
    Pack solvent molecules and ions into voids at a given volume defined by a geometry.
    The packing is performed by packmol.

    :param atoms: ase Atoms object that specifies particles that solvent is to be packed around. The packed solvent molecules will be added to this atoms object.
    :type atoms: Atoms object
    :param solvent: ase Atoms object that specify the solvent.
    :type solvent: Atoms object
    :param ions: a list of ase atoms object to specify the ions.
    :type: list of ase Atoms object.
    :param nummol: Number of solvent molecules
    :type nummol: int
    :param numions: as list of integers with number of ions to be placed.
    :type numions: list of int.
    :param volume: Void volume in :math:`Å^3` to be filled. Can only be used if `nummol=None`.
    :type volume: float
    :param density: solvent density. Used to compute the number of solvent molecules to be packed if `volume` is provided.
    :type density: float
    :param geometry: Geometry object specifying where to pack solvent and ions
    :type geometry: Geometry object
    :param side: Pack solvent inside/outside of geometry
    :type side: str
    :param pbc: Inner margin to add to the simulation box to avoid overlapping atoms over periodic boundary conditions. This is necessary because packmol doesn't support periodic boundary conditions.
    :type pbc: float or array_like
    :param tolerance: Minimum separation distance between molecules.
    :type tolerance: float

    :returns: Coordinates of the packed solvent and or ions
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

        
    if (volume is None and nummol is None):
        raise ValueError("Either volume or the number of molecules needed")
    elif (volume is not None) and (nummol is not None):
        raise ValueError("Either volume or the number of molecules needed")

    if volume is not None:

        MMsolv = sum(solvent.get_masses())
        numol_p_V = get_molecules_per_volume(density,MMsolv)
        nummol = volume*numol_p_V
        nummol = int(np.round(nummol))
    
    if not(isinstance(ions, list)) and ions is not None:
        ions = [ions]

    format_s, format_v = "pdb", "proteindatabank"
    side += "side"
    if ions is not None and numions is None:
        raise ValueError("Either number of ions to be packed.")

    if atoms is None and geometry is None:
        raise ValueError("Either atoms or geometry has to be given")
    elif geometry is None:
        # The default geometry to pack the solvent is a box which capsules the solid
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

        # Copy solvent.pdb to temporary directory
        if solvent is not None and len(solvent) > 0:
            solvent.write(f"solvent.{format_s}", format=format_v)

        if ions is not None and len(ions)>0:
            # Copy each ions.pdb in the list of ions to temporary directory
            for i in range(len(ions)):
                ions[i].write(f"ion_{i}.{format_s}",format=format_v)
        for i in range(len(numions)):
            if not(isinstance(numions[i],int)):
                    numions[i] = int(numions[i])
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
            if ions is not None and len(ions)>0:
                f.write(geometry.packmol_structure_ions(numions,side))
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
            solvent_ions = ase.io.read(f"{tmp_dir}/out.{format_s}", format=format_v)
            solvent_ions.set_pbc(True)
        except FileNotFoundError as e: 
            raise FileNotFoundError(f"Packmol output not found\nOutput from Packmol: {out}\nError from Packmol: {err}\n{e}")

    os.chdir(cwd)
    if atoms is None:
        solvent_ions.set_cell(cell)
    else:
        #remove solid
        if len(atoms) > 0:
            del solvent_ions[:len(atoms)]
        solvent_ions.set_cell(cell)
        atoms += solvent_ions

    return solvent_ions



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
    with open("solvents.pkl","rb") as f:
        solvents_pkl = pickle.load(f)

    with open("ions.pkl","rb") as f:
        ions_pkl = pickle.load(f)

    i_solvent = np.random.randint(1,8)
    solvent_name = solvents_pkl[str(i_solvent)]["Name"]
    solvent = solvents_pkl[str(i_solvent)]["Atoms"]
    density_solvent = solvents_pkl[str(i_solvent)]["density"]
    symbol_solv = solvent.get_chemical_formula()
    sovent_box_cell = solvent
    z = 8
    adsorbate =adsorbate.atoms
    adsorbate.set_cell([cell[0],cell[1],[0.,0.,z]])
    i_ions= np.random.randint(1,8)
    ion1 = ions_pkl[i_ions]["Atoms"]
    ion1_name = ions_pkl[i_ions]["Atoms"].get_chemical_formula()
    i_ions= np.random.randint(1,8)
    ion2 = ions_pkl[i_ions]["Atoms"]
    ion2_name = ions_pkl[i_ions]["Atoms"].get_chemical_formula()

    solvent_box_cell = pack_solvent(volume=adsorbate.get_volume(),atoms=adsorbate,solvent=solvent,ions=[ion1,ion2],numions=[1,1.0],density=density_solvent)
    a = atoms.positions[:, 2].argmax()
    height = 3.0

    max_z = atoms.positions[a, 2] + height

    solvent_box_cell.translate([0,0,max_z])

    interface = atoms + solvent_box_cell

    interface.center(vacuum=15, axis=2) 

    interface.wrap()
    interface.write('{0}-interface-{1}-{2}-n-{3}.POSCAR'.format(symbol,solvent_name,ion1_name,ion2_name))
    print('{0}-interface-{1}-{2}-n-{3}.POSCAR'.format(symbol,solvent_name,ion1_name,ion2_name))

    


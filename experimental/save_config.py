"""
Given a bulk and an adsorbate, this helper script generates all the configurations
of that adsorbate on various surfaces of that bulk.
"""
import argparse
import lmdb
import os
import pickle
from ocdata.adsorptions import *
from ocdata.base_atoms.pkls import BULK_PKL, ADSORBATE_PKL
from ocpmodels.preprocessing import AtomsToGraphs
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm 

def generate_all_structures(bulk_atoms, adsorbate=''):
    """
    Args:
        bulk_atoms:         The atom object of a bulk. 
        adsorbate:          The SMILE string of the adsorbate

    Returns:
        adsorbed_surfaces   The list of atoms objects. Each object is 
                            an adsorbate on a surface. 
    """
    all_surfaces = enumerate_surfaces(bulk_atoms)
    adsorbed_surfaces = enumerate_adslabs(bulk_atoms, all_surfaces, adsorbate)
    return adsorbed_surfaces

def enumerate_adslabs(bulk_atoms, surface_list, adsorbate, constrained=True):
    # get the adsorbate from the database
    with open(ADSORBATE_PKL, 'rb') as f:
        inv_index = pickle.load(f)
    adsorbate_atoms, smiles, bond_indices = [ads for idx, ads in inv_index.items() if ads[1] == adsorbate][0]
    
    # place adsorbate onto all surfaces and place them into a dictionary,
    # where the keys are surface info (miller, shift, top)
    adslabs_configurations = {}
    for surface in surface_list:
        surface_struct, millers, shift, top = surface
        unit_surface_atoms = AseAtomsAdaptor.get_atoms(surface_struct)
        surface_atoms = tile_atoms(unit_surface_atoms)
        tag_surface_atoms(bulk_atoms, surface_atoms)
        adsorbed_surface = add_adsorbate_onto_surface(surface_atoms, adsorbate_atoms, bond_indices, sample=False)
        # add constraints
        if constrained:
            adsorbed_surface = [constrain_surface(adslab) for adslab in adsorbed_surface]
        adslabs_configurations[(millers, shift, top)] = adsorbed_surface
    return adslabs_configurations


if __name__ == "__main__":
#     # Parse a few arguments.
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--composition",
#         type=int, 
#         default=None,
#         required=True,
#         help="composition_tags (1/2/3 for unary / binary / ternary)",
#     )
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--adsorbate",
#         type=str, 
#         default=None,
#         required=True,
#         help="The smile string of the adsorbate",
#     )
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--db_path",
#         type=str, 
#         default=None,
#         required=True,
#         help="The relative path of the",
#     )
    
#     args = parser.parse_args()
    
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_distances=True,
        r_fixed=True,
    )
    
    # Initialize lmdb paths
#     db_path = os.path.join(args.db_path, "data.lmdb")
    db_path = "data.lmdb"
    db = lmdb.open(
            db_path,
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
    )
    
    # decide on what compositions
    with open(BULK_PKL, 'rb') as f:
        inv_index = pickle.load(f)
    bulks = inv_index[1]
#     bulks = inv_index[args.composition]

    
    with db.begin(write=True) as txn:
        for idx, bulk_info in tqdm(enumerate(bulks)):
            bulk_atoms, mpid = bulk_info
            try:
                configurations = generate_all_structures(bulk_atoms, adsorbate='*OH')
                for surface, config_list in configurations.items():
                    dl = a2g.convert_all(config_list, disable_tqdm=True)
                    txn.put(f"{idx}".encode("ascii"), pickle.dumps(dl[0], protocol=-1))
            except ValueError:
                continue
    db.sync()
    db.close()

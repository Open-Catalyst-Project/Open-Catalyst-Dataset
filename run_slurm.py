import argparse
import logging
import pickle

from sample_structure import StructureSampler

def parse_args():
    parser = argparse.ArgumentParser(description='Sample adsorbate and bulk surface(s)')

    parser.add_argument('--indices_file', type=str, required=True, help='Pickle containing tuples of (bulk mpid, adsorbate smiles, surface index)')
    parser.add_argument('--file_row_index', type=int, default=None, help='Specify one row of the file to run')
    parser.add_argument('--adsorbate_index_mapping', type=str, required=True, help='Text file that maps index to adsorbate')
    parser.add_argument('--mpid_index_mapping', type=str, required=True, help='Text file that maps index to mpid')

    ########## These args are used for sample_structure.py ##########
    parser.add_argument('--bulk_db', type=str, required=True, help='Underlying db for bulks')
    parser.add_argument('--adsorbate_db', type=str, required=True, help='Underlying db for adsorbates')
    parser.add_argument('--output_dir', type=str, required=True, help='Root directory for outputs')
    parser.add_argument('--precomputed_structures', type=str, default=None, help='Root directory of precomputed structures')
    parser.add_argument('--verbose', action='store_true', default=False, help='Log detailed info')

    args = parser.parse_args()
    return args

def invert_mappings(filename, expected_elems):
    # takes a file that maps index to either mpid or smiles, and returns the inverse dict
    # that maps mpid/smiles to bulk/adsorbate indices
    # expected_elems is how many space-separated items you expect to see each line
    str_to_ind = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            elems = line.strip().split(' ')
            assert len(elems) == expected_elems
            str_to_ind[elems[1]] = int(elems[0])
    return str_to_ind

def run_sample_structure(args, adsorbate, bulk, surface):
    print(f'Running sample_structure.py for adsorbate {adsorbate}, bulk {bulk}, surface {surface}')

    # manually set some args
    args.enumerate_all_structures = True
    args.adsorbate_index = adsorbate
    args.bulk_indices = str(bulk)
    args.surface_index = surface

    job = StructureSampler(args)
    job.run()

if __name__ == '__main__':
    args = parse_args()

    # get mappings from string -> int
    smiles_to_ind = invert_mappings(args.adsorbate_index_mapping, 2)
    mpid_to_ind = invert_mappings(args.mpid_index_mapping, 3)

    with open(args.indices_file, 'rb') as f:
        all_structures = pickle.load(f)

    print(f'Found: {len(all_structures)} structures in file')

    if args.file_row_index is not None:
        all_structures = [all_structures[args.file_row_index]]
        printf(f'Only running line {args.file_row_index}')

    for structure_tuple in all_structures:
        mpid, smiles, surface_ind = structure_tuple
        run_sample_structure(args, smiles_to_ind[smiles], mpid_to_ind[mpid], surface_ind)

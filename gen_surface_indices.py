import argparse
import logging
import pickle

from ocdata.bulk_obj import Bulk
from run_slurm import invert_mappings

'''
Given a file that's a pickled list of (bulk mpid, adsorbate smiles),
generate a file that's a pickled list of (bulk mpid, adsorbate smiles, surface index)
for all possible surfaces for that bulk.
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Sample adsorbate and bulk surface(s)')

    parser.add_argument('--indices_file', type=str, required=True, help='Pickle containing tuples of (bulk, adsorbate) strings')
    parser.add_argument('--output_file', type=str, required=True, help='Output filename for pickle of (bulk, adsorbate, surface index)')

    parser.add_argument('--bulk_db', type=str, required=True, help='Underlying db for bulks')
    parser.add_argument('--mpid_index_mapping', type=str, required=True, help='Text file that maps index to mpid')
    parser.add_argument('--precomputed_structures', type=str, default=None, help='Root directory of precomputed structures')

    # check that all needed args are supplied
    args = parser.parse_args()
    return args

def count_possible_surfaces(bulk_ind, bulk_by_ind, precomputed_structures, mpid):
    # given a bulk id (int), return the number of possible surfaces (int)
    # also verifies that the mpid is matching in bulk_db
    bulk = Bulk(bulk_by_ind, precomputed_structures, bulk_ind)
    assert bulk.mpid == mpid, f'expected {mpid}, found {bulk.mpid}'
    return(len(bulk.get_possible_surfaces()))

if __name__ == '__main__':
    args = parse_args()

    mpid_to_ind = invert_mappings(args.mpid_index_mapping, 3)

    with open(args.indices_file, 'rb') as f:
        all_inputs = pickle.load(f) # list of (mpid, smiles) tuples

    all_outputs = [] # list of (mpid, smiles, surface index) tuples
    with open(args.bulk_db, 'rb') as f:
        bulk_by_ind = pickle.load(f)

    for structure_tuple in all_inputs:
        assert len(structure_tuple) == 2
        mpid, smiles = structure_tuple
        num_surfaces = count_possible_surfaces(mpid_to_ind[mpid], bulk_by_ind, args.precomputed_structures, mpid)
        print(f'found {num_surfaces} possible surfaces for {mpid}, {smiles}')
        for ind in range(num_surfaces):
            all_outputs.append((mpid, smiles, ind))

    with open(args.output_file, 'wb') as outf:
        pickle.dump(all_outputs, outf)

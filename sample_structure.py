
from ocdata.vasp import run_vasp, write_vasp_input_files
from ocdata.adsorbates import Adsorbate
from ocdata.surfaces import Surface
from ocdata.combined import Combined

import argparse
import math
import numpy as np
import os
import pickle
import time

def choose_n_elems(n_cat_elems_weights={1: 0.05, 2: 0.65, 3: 0.3}):
    '''
    Chooses the number of species we should look for in this sample.

    Arg:
        n_cat_elems_weights A dictionary whose keys are integers containing the
                            number of species you want to consider and whose
                            values are the probabilities of selecting this
                            number. The probabilities must sum to 1.
    Returns:
        n_elems             An integer showing how many species have been chosen.
        sampling_string     Enum string of [chosen n_elem]/[total number of choices]
    '''

    n_elems = list(n_cat_elems_weights.keys())
    weights = list(n_cat_elems_weights.values())
    assert math.isclose(sum(weights), 1)

    n_elem = np.random.choice(n_elems, p=weights)
    sampling_string = str(n_elem) + "/" + str(len(n_elems))
    return n_elem, sampling_string

class StructureSampler():
    def __init__(self, args):
        # set up args, random seed, directories
        self.args = args

        np.random.seed(self.args.seed)
        output_name_template = f'random{self.args.seed}'
        self.bulk_dir = os.path.join(self.args.output_dir, output_name_template, 'surface')
        self.adsorbed_bulk_dir = os.path.join(self.args.output_dir, output_name_template, 'adsorbed_surface')

        # todo all combos

    def load_adsorbate(self):
        # sample a random adsorbate, or load the specified one
        if self.args.adsorbate_index: # can make this multiple indices in the future
            self.adsorbate = Adsorbate(self.args.adsorbate_index, self.args.adsorbate_db)
        else:
            self.adsorbate = Adsorbate(self.args.adsorbate_db)

    def load_surfaces(self):
        # todo: make a list of surfaces
        n_elems, elem_sampling_str = choose_n_elems() # todo make weights an input param
        self.surface = Surface(self.args.bulk_db, n_elems, elem_sampling_str, self.args.precomputed_structures)


    def run(self):

        start = time.time()

        self.load_adsorbate()
        self.load_surfaces()

        # todo: loop through surfaces
        combined = Combined(self.adsorbate, self.surface)

        bulk_dict = self.surface.get_bulk_dict()
        adsorbed_bulk_dict = combined.get_adsorbed_bulk_dict()

        write_vasp_input_files(bulk_dict["bulk_atomsobject"], self.bulk_dir)
        write_vasp_input_files(adsorbed_bulk_dict["adsorbed_bulk_atomsobject"], self.adsorbed_bulk_dir)

        self.write_metadata_pkl(bulk_dict, os.path.join(self.bulk_dir, 'metadata.pkl'))
        self.write_metadata_pkl(adsorbed_bulk_dict, os.path.join(self.adsorbed_bulk_dir, 'metadata.pkl'))

        end = time.time()
        print(f'Done ({round(end - start, 2)}s)! Seed = {self.args.seed}')

    def write_metadata_pkl(self, dict_to_write, path):
        file_path = os.path.join(path, 'metadata.pkl')
        with open(path, 'wb') as f:
            pickle.dump(dict_to_write, f)

'''
currently testing with:
python sample_structure.py --bulk_db ocdata/base_atoms/pkls/bulks_may12.pkl --adsorbate_db ocdata/base_atoms/pkls/adsorbates.pkl \
--precomputed_structures /private/home/sidgoyal/Open-Catalyst-Dataset/ocdata/precomputed_structure_info --output_dir junk --seed 1
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Sample adsorbate and bulk surface(s)')

    parser.add_argument('--seed', type=int, default=None, help='Random seed for sampling')

    # input and output
    parser.add_argument('--bulk_db', type=str, required=True, help='Underlying db for bulks')
    parser.add_argument('--adsorbate_db', type=str, required=True, help='Underlying db for adsorbates')
    parser.add_argument('--output_dir', type=str, required=True, help='Output dir path')

    # for optimized (automatically try to use optimized if this is provided)
    parser.add_argument('--precomputed_structures', type=str, default=None, help='Root directory of precomputed structures')

    # enumerating all combinations:
    parser.add_argument('--enumerate_all_structures', action='store_true', default=False,
        help='Find all possible structures given a specific adsorbate and a list of bulks')
    parser.add_argument('--adsorbate_index', type=int, default=None, help='adsorbate index (int)')
    # todo need to provide num elems?
    parser.add_argument('--bulk_indices', type=str, default=None, help='Comma separated list of bulk indices') # TODO make file later

    args = parser.parse_args()
    if args.enumerate_all_structures:
        if args.adsorbate_index is None or args.bulk_indices is None:
            parser.error('Enumerating all structures requires adsorbate index and bulk indices file')
    else:
        if args.seed is None:
            parser.error('Seed is required when sampling one random structure')
    return args


if __name__ == '__main__':
    args = parse_args()
    job = StructureSampler(args)
    job.run()
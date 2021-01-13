
from ocdata.vasp import run_vasp, write_vasp_input_files
from ocdata.adsorbates import Adsorbate
from ocdata.bulk_obj import Bulk
from ocdata.surfaces import Surface
from ocdata.combined import Combined

import argparse
import logging
import math
import numpy as np
import os
import pickle
import time


class StructureSampler():
    def __init__(self, args):
        # set up args, random seed, directories
        self.args = args
        if self.args.bulk_indices:
            self.bulk_indices_list = self.args.bulk_indices.split(',')

        self.logger = logging.getLogger()
        logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S')
        if self.args.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

        np.random.seed(self.args.seed)

    def load_adsorbate(self):
        # sample a random adsorbate, or load the specified one
        # stores it in self.adsorbate
        if self.args.enumerate_all_structures: # can make this multiple indices in the future
            self.adsorbate = Adsorbate(self.args.adsorbate_db, self.args.adsorbate_index)
        else:
            self.adsorbate = Adsorbate(self.args.adsorbate_db)

    def load_bulks(self):
        '''
        Loads bulk structures (one random or a list of specified ones)
        and stores them in self.all_bulks
        '''
        self.all_bulks = []
        with open(self.args.bulk_db, 'rb') as f:
            inv_index = pickle.load(f)

        if self.args.enumerate_all_structures:
            for ind in self.bulk_indices_list:
                self.all_bulks.append(Bulk(inv_index, self.args.precomputed_structures, int(ind)))
        else:
            self.all_bulks.append(Bulk(inv_index, self.args.precomputed_structures))

    def load_and_write_surfaces(self):
        '''
        Loops through all bulks and chooses one random or all possible surfaces;
        writes info for that surface and combined surface+adsorbate
        '''
        for bulk_ind, bulk in enumerate(self.all_bulks):
            possible_surfaces = bulk.get_possible_surfaces()
            if self.args.enumerate_all_structures:
                self.logger.info(f'Enumerating all {len(possible_surfaces)} surfaces for bulk {self.bulk_indices_list[bulk_ind]}')
                for surface_ind, surface_info in enumerate(possible_surfaces):
                    surface = Surface(bulk, surface_info, surface_ind, len(possible_surfaces))
                    self.combine_and_write(surface, self.bulk_indices_list[bulk_ind], surface_ind)
            else:
                surface_info_index = np.random.choice(len(possible_surfaces))
                surface = Surface(bulk, possible_surfaces[surface_info_index], surface_info_index, len(possible_surfaces))
                self.combine_and_write(surface)

    def combine_and_write(self, surface, cur_bulk_index=None, cur_surface_index=None):
        # writes output files for the surface itself and the surface+adsorbate
        combined = Combined(self.adsorbate, surface)

        bulk_dict = surface.get_bulk_dict()
        adsorbed_bulk_dict = combined.get_adsorbed_bulk_dict()

        if self.args.enumerate_all_structures:
            output_name_template = f'{self.args.adsorbate_index}_{cur_bulk_index}_surface{cur_surface_index}'
        else:
            output_name_template = f'random{self.args.seed}'
        bulk_dir = os.path.join(self.args.output_dir, output_name_template, 'surface')
        adsorbed_bulk_dir = os.path.join(self.args.output_dir, output_name_template, 'adsorbed_surface')

        write_vasp_input_files(bulk_dict['bulk_atomsobject'], bulk_dir)
        write_vasp_input_files(adsorbed_bulk_dict['adsorbed_bulk_atomsobject'], adsorbed_bulk_dir)
        self.logger.info(f"wrote surface ({bulk_dict['bulk_samplingstr']}) to {bulk_dir}")
        self.logger.info(f"wrote adsorbed surface ({adsorbed_bulk_dict['adsorbed_bulk_samplingstr']}) to {adsorbed_bulk_dir}")

        self.write_metadata_pkl(bulk_dict, os.path.join(bulk_dir, 'metadata.pkl'))
        self.write_metadata_pkl(adsorbed_bulk_dict, os.path.join(adsorbed_bulk_dir, 'metadata.pkl'))

    def write_metadata_pkl(self, dict_to_write, path):
        file_path = os.path.join(path, 'metadata.pkl')
        with open(path, 'wb') as f:
            pickle.dump(dict_to_write, f)

    def run(self):

        start = time.time()

        self.load_adsorbate()
        self.load_bulks()
        self.load_and_write_surfaces()

        end = time.time()
        print(f'Done! ({round(end - start, 2)}s)')

def parse_args():
    parser = argparse.ArgumentParser(description='Sample adsorbate and bulk surface(s)')

    parser.add_argument('--seed', type=int, default=None, help='Random seed for sampling')

    # input and output
    parser.add_argument('--bulk_db', type=str, required=True, help='Underlying db for bulks')
    parser.add_argument('--adsorbate_db', type=str, required=True, help='Underlying db for adsorbates')
    parser.add_argument('--output_dir', type=str, required=True, help='Output dir path')

    # for optimized (automatically try to use optimized if this is provided)
    parser.add_argument('--precomputed_structures', type=str, default=None, help='Root directory of precomputed structures')

    # required args for enumerating all combinations:
    parser.add_argument('--enumerate_all_structures', action='store_true', default=False,
        help='Find all possible structures given a specific adsorbate and a list of bulks')
    parser.add_argument('--adsorbate_index', type=int, default=None, help='adsorbate index (int)')
    parser.add_argument('--bulk_indices', type=str, default=None, help='Comma separated list of bulk indices') # TODO change to file later

    parser.add_argument('--verbose', action='store_true', default=False, help='Log detailed info')

    args = parser.parse_args()
    if args.enumerate_all_structures:
        if args.adsorbate_index is None or args.bulk_indices is None:
            parser.error('Enumerating all structures requires adsorbate index and bulk indices file')
    elif args.seed is None:
            parser.error('Seed is required when sampling one random structure')
    return args


if __name__ == '__main__':
    args = parse_args()
    job = StructureSampler(args)
    job.run()
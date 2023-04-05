import argparse
import logging
import os
import pickle
import time

import numpy as np

from ocdata.utils.vasp import write_vasp_input_files

from ocdata.core import Adsorbate, Bulk, Slab, AdsorbateSlabConfig

'''
test commands

python structure_generator.py \
--bulk_db databases/pkls/bulks.pkl \
--adsorbate_db databases/pkls/adsorbates.pkl \
--precomputed_slabs_dir /checkpoint/janlan/ocp/input_dbs/precomputed_surfaces_2021Sep20/ \
--adsorbate_index 1 --bulk_index 0 --surface_index 0 \
--output_dir junktest/ \
--random_placements --random_sites 10 --num_augmentations 2 --heuristic_placements \
--no_vasp
'''

class StructureGenerator:
    """
    A class that creates adsorbate/bulk/surface objects and
    writes vasp input files for one of the following options:
    - all placements for one specified adsorbate, one specified bulk, one specified surface
        (random, heuristic, or both)
    - optional extensions: loop over a list of adsorbates

    The output directory structure will look like the following:
    - For sampling a random structure, the directories will be `random{seed}/surface` and
        `random{seed}/adslab` for the surface alone and the adsorbate+surface, respectively.
    - For enumerating all structures, the directories will be `{adsorbate}_{bulk}_{surface}/surface`
        and `{adsorbate}_{bulk}_{surface}/adslab{config}`, where everything in braces are the
        respective indices.

    Attributes
    ----------
    args : argparse.Namespace
        contains all command line args
    logger : logging.RootLogger
        logging class to print info
    adsorbate : Adsorbate
        the selected adsorbate object
    all_bulks : list
        list of `Bulk` objects
    bulk_indices_list : list
        list of specified bulk indices (ints) that we want to select

    Public methods
    --------------
    run()
        selects the appropriate materials and writes to files
    """

    def __init__(self, args):
        """
        Set up args from argparse, random seed, and logging.
        """
        self.args = args

        self.logger = logging.getLogger()
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
        self.logger.setLevel(logging.INFO if self.args.verbose else logging.WARNING)

        self.logger.info(
            f"Processing adsorbate {self.args.adsorbate_index}, bulk {self.args.bulk_index}, surface {self.args.surface_index}"
        )
        if self.args.seed:
            np.random.seed(self.args.seed)

    def run(self):
        """
        Generates adsorbate/bulk/surface objects and writes to files.
        """
        start = time.time()

        # create adsorbate, bulk, and surface objects
        self.bulk = Bulk(bulk_id_from_db=self.args.bulk_index, bulk_db_path=self.args.bulk_db)
        self.adsorbate = Adsorbate(adsorbate_id_from_db=self.args.bulk_index, adsorbate_db_path=self.args.adsorbate_db)
        all_slabs = self.bulk.get_slabs(max_miller=self.args.max_miller, precomputed_slabs_dir=self.args.precomputed_slabs_dir)
        self.slab = all_slabs[self.args.surface_index]

        # create adslabs
        self.rand_adslabs, self.heur_adslabs = None, None
        if self.args.heuristic_placements:
            self.heur_adslabs = AdsorbateSlabConfig(self.slab, self.adsorbate, 
                num_augmentations_per_site=self.args.num_augmentations, mode="heuristic")
        if self.args.random_placements:
            self.rand_adslabs = AdsorbateSlabConfig(self.slab, self.adsorbate, 
                self.args.random_sites, self.args.num_augmentations, mode="random")

        self.write_surface()
        if self.heur_adslabs:
            self.write_adslabs(self.heur_adslabs, "heur")
        if self.rand_adslabs:
            self.write_adslabs(self.rand_adslabs, "rand")

        end = time.time()
        self.logger.info(f"Done! ({round(end - start, 2)}s)")

    def write_surface(self):
        """
        outputdir/
            bulk0/
                surface0/
                    surface/POSCAR
                    heur0/POSCAR
                    heur1/POSCAR
                    rand0/POSCAR
                    ...
                surface1/
                    ...
            bulk1/
                ...
        """

        os.makedirs(os.path.join(self.args.output_dir, f"bulk{self.args.bulk_index}"),
            exist_ok=True)
        os.makedirs(os.path.join(self.args.output_dir, f"bulk{self.args.bulk_index}",
            f"surface{self.args.surface_index}"), exist_ok=True)

        # write vasp files
        slab_alone_dir = os.path.join(self.args.output_dir, f"bulk{self.args.bulk_index}",
            f"surface{self.args.surface_index}", "surface")
        if not os.path.exists(os.path.join(slab_alone_dir, "POSCAR")):
            # skip surface if already written.
            # this happens when we process multiple adsorbates per surface
            write_vasp_input_files(self.slab.atoms, slab_alone_dir)
            # we don't take out files for self.args.no_vasp because
            # we do want to run vasp on these surface files

        # write metadata
        metadata_path = os.path.join(self.args.output_dir, f"bulk{self.args.bulk_index}",
            f"surface{self.args.surface_index}", "surface", "metadata.pkl")
        if not os.path.exists(metadata_path):
            metadata_dict = self.slab.get_metadata_dict()
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata_dict, f)

    def write_adslabs(self, adslab_obj, mode_str):
        # write adslabs, either random or heuristic
        # see dir structure in write_surface()
        for adslab_ind, adslab_atoms in enumerate(adslab_obj.atoms_list):
            adslab_dir = os.path.join(self.args.output_dir, f"bulk{self.args.bulk_index}",
                f"surface{self.args.surface_index}", f"{mode_str}{adslab_ind}")

            # vasp files
            write_vasp_input_files(adslab_atoms, adslab_dir)
            if self.args.no_vasp:
                # a bit hacky but ASE defaults to writing everything out
                for unused_file in ["KPOINTS", "INCAR", "POTCAR"]:
                    os.remove(os.path.join(adslab_dir, unused_file))

            # write dict for metadata
            metadata_path = os.path.join(self.args.output_dir, f"bulk{self.args.bulk_index}",
                f"surface{self.args.surface_index}", f"{mode_str}{adslab_ind}", "metadata.pkl")
            metadata_dict = adslab_obj.get_metadata_dict(adslab_ind)
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata_dict, f)
    

def parse_args():
    parser = argparse.ArgumentParser(description="Sample adsorbate and bulk surface(s)")

    # input databases
    parser.add_argument(
        "--bulk_db", type=str, required=True, help="Underlying db for bulks (.pkl)"
    )
    parser.add_argument(
        "--adsorbate_db", type=str, required=True, help="Underlying db for adsorbates (.pkl)"
    )
    # for optimized (automatically try to use optimized if this is provided)
    parser.add_argument(
        "--precomputed_slabs_dir",
        type=str,
        default=None,
        help="Root directory of precomputed surfaces",
    )

    # material specifications
    parser.add_argument(
        "--adsorbate_index", type=int, default=None, help="Adsorbate index (int)"
    )
    parser.add_argument(
        "--bulk_index", type=int, default=None, help="Bulk index (int)",
    )
    parser.add_argument(
        "--surface_index", type=int, default=None, help="Surface index (int)"
    )

    # output
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Root directory for outputs"
    )

    # other options
    parser.add_argument(
        "--max_miller",
        type=int,
        default=2,
        help="Max miller indices to consider for generating surfaces",
    )
    parser.add_argument(
        "--random_placements",
        action="store_true",
        default=False,
        help="Generate random placements",
    )
    parser.add_argument(
        "--heuristic_placements",
        action="store_true",
        default=False,
        help="Generate heuristic placements",
    )
    parser.add_argument(
        "--random_sites",
        type=int,
        default=None,
        help="Number of random placement per adsorbate/surface if args.random_placements is set to True",
    )
    parser.add_argument(
        "--num_augmentations",
        type=int,
        default=1,
        help="Number of random augmentations (i.e. rotations) per site",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling/random sites generation",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Log detailed info"
    )
    parser.add_argument(
        "--no_vasp",
        action="store_true",
        default=False,
        help="Do not write out POTCAR/INCAR/KPOINTS for adslabs",
    )

    args = parser.parse_args()

    # check that all needed args are supplied
    if not (args.random_placements or args.heuristic_placements):
        parser.error("Must choose either or both of random or heuristic placements")
    if args.random_placements and (args.random_sites is None or args.random_sites <= 0):
        parser.error("Must specify number of sites for random placements")

    return args


if __name__ == "__main__":
    # This handles one adsorbate-surface material
    # to handle files, write a different wrapper that reads in lines
    # and manually sets the args.*_index variables
    args = parse_args()
    job = StructureGenerator(args)
    job.run()
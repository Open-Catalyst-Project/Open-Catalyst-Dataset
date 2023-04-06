import argparse
import logging
import multiprocessing as mp
import os
import pickle
import time

import numpy as np

from ocdata.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab
from ocdata.utils.vasp import write_vasp_input_files


class StructureGenerator:
    """
    A class that creates adsorbate/bulk/slab objects given specified indices,
    and writes vasp input files and metadata for multiple placements of the adsorbate
    on the slab. You can choose random, heuristic, or both types of placements.

    The output directory structure will have the following nested structure,
    where "files" represents the vasp input files and the metadata.pkl:
        outputdir/
            bulk0/
                surface0/
                    surface/files
                    ads0/
                        heur0/files
                        heur1/files
                        rand0/files
                        ...
                    ads1/
                        ...
                surface1/
                    ...
            bulk1/
                ...

    Arguments
    ----------
    args: argparse.Namespace
        Contains all command line args
    bulk_index: int
        Index of the bulk within the bulk db
    surface_index: int
        Index of the surface in the list of all possible surfaces
    adsorbate_index: int
        Index of the adsorbate within the adsorbate db
    """

    def __init__(self, args, bulk_index, surface_index, adsorbate_index):
        """
        Set up args from argparse, random seed, and logging.
        """
        self.args = args
        self.bulk_index = bulk_index
        self.surface_index = surface_index
        self.adsorbate_index = adsorbate_index

        self.logger = logging.getLogger()
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
        self.logger.setLevel(logging.INFO if self.args.verbose else logging.WARNING)

        self.logger.info(
            f"Starting adsorbate {self.adsorbate_index}, bulk {self.bulk_index}, surface {self.surface_index}"
        )
        if self.args.seed:
            np.random.seed(self.args.seed)

    def run(self):
        """
        Create adsorbate/bulk/surface objects, generate adslab placements,
        and write to files.
        """
        start = time.time()

        # create adsorbate, bulk, and surface objects
        self.bulk = Bulk(
            bulk_id_from_db=self.bulk_index, bulk_db_path=self.args.bulk_db
        )
        self.adsorbate = Adsorbate(
            adsorbate_id_from_db=self.adsorbate_index,
            adsorbate_db_path=self.args.adsorbate_db,
        )
        all_slabs = self.bulk.get_slabs(
            max_miller=self.args.max_miller,
            precomputed_slabs_dir=self.args.precomputed_slabs_dir,
        )
        self.slab = all_slabs[self.surface_index]

        # create adslabs
        self.rand_adslabs, self.heur_adslabs = None, None
        if self.args.heuristic_placements:
            self.heur_adslabs = AdsorbateSlabConfig(
                self.slab,
                self.adsorbate,
                num_augmentations_per_site=self.args.num_augmentations,
                mode="heuristic",
            )
        if self.args.random_placements:
            self.rand_adslabs = AdsorbateSlabConfig(
                self.slab,
                self.adsorbate,
                self.args.random_sites,
                self.args.num_augmentations,
                mode="random",
            )

        # write files
        self._write_surface()
        if self.heur_adslabs:
            self._write_adslabs(self.heur_adslabs, "heur")
        if self.rand_adslabs:
            self._write_adslabs(self.rand_adslabs, "rand")

        end = time.time()
        self.logger.info(
            f"Completed adsorbate {self.adsorbate_index}, bulk {self.bulk_index}, surface {self.surface_index} ({round(end - start, 2)}s)"
        )

    def _write_surface(self):
        """
        Writes vasp inputs and metadata for the slab alone
        """

        os.makedirs(
            os.path.join(self.args.output_dir, f"bulk{self.bulk_index}"), exist_ok=True
        )
        os.makedirs(
            os.path.join(
                self.args.output_dir,
                f"bulk{self.bulk_index}",
                f"surface{self.surface_index}",
            ),
            exist_ok=True,
        )

        # write vasp files
        slab_alone_dir = os.path.join(
            self.args.output_dir,
            f"bulk{self.bulk_index}",
            f"surface{self.surface_index}",
            "surface",
        )
        if not os.path.exists(os.path.join(slab_alone_dir, "POSCAR")):
            # Skip surface if already written;
            # this happens when we process multiple adsorbates per surface.
            write_vasp_input_files(self.slab.atoms, slab_alone_dir)

        # write metadata
        metadata_path = os.path.join(
            self.args.output_dir,
            f"bulk{self.bulk_index}",
            f"surface{self.surface_index}",
            "surface",
            "metadata.pkl",
        )
        if not os.path.exists(metadata_path):
            metadata_dict = self.slab.get_metadata_dict()
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata_dict, f)

    def _write_adslabs(self, adslab_obj, mode_str):
        """
        Write one set of adslabs (called separately for random and heurstic placements)
        """
        for adslab_ind, adslab_atoms in enumerate(adslab_obj.atoms_list):
            adslab_dir = os.path.join(
                self.args.output_dir,
                f"bulk{self.bulk_index}",
                f"surface{self.surface_index}",
                f"ads{self.adsorbate.adsorbate_id_from_db}",
                f"{mode_str}{adslab_ind}",
            )

            # vasp files
            write_vasp_input_files(adslab_atoms, adslab_dir)
            if self.args.no_vasp:
                # A bit hacky but ASE defaults to writing everything out.
                for unused_file in ["KPOINTS", "INCAR", "POTCAR"]:
                    os.remove(os.path.join(adslab_dir, unused_file))

            # write dict for metadata
            metadata_path = os.path.join(adslab_dir, "metadata.pkl")
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
        "--adsorbate_db",
        type=str,
        required=True,
        help="Underlying db for adsorbates (.pkl)",
    )
    # for optimized (automatically try to use optimized if this is provided)
    parser.add_argument(
        "--precomputed_slabs_dir",
        type=str,
        default=None,
        help="Root directory of precomputed surfaces",
    )

    # material specifications, option A: provide one set of indices
    parser.add_argument(
        "--adsorbate_index", type=int, default=None, help="Adsorbate index (int)"
    )
    parser.add_argument(
        "--bulk_index",
        type=int,
        default=None,
        help="Bulk index (int)",
    )
    parser.add_argument(
        "--surface_index", type=int, default=None, help="Surface index (int)"
    )

    # material specifications, option B: provide one set of indices
    parser.add_argument(
        "--indices_file",
        type=str,
        default=None,
        help="File containing adsorbate_bulk_surface indices",
    )
    parser.add_argument(
        "--chunk_index", type=int, default=None, help="Row(s) in file to run"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=1, help="Chunk size for parallelization"
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
        "--no_vasp",
        action="store_true",
        default=False,
        help="Do not write out POTCAR/INCAR/KPOINTS for adslabs",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Log detailed info"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of workers for multiprocessing when given a file of indices",
    )

    args = parser.parse_args()

    # check that all needed args are supplied
    if not (args.random_placements or args.heuristic_placements):
        parser.error("Must choose either or both of random or heuristic placements")
    if args.random_placements and (args.random_sites is None or args.random_sites <= 0):
        parser.error("Must specify number of sites for random placements")
    if not args.indices_file:
        if (
            args.adsorbate_index is None
            or args.bulk_index is None
            or args.surface_index is None
        ):
            parser.error("Must provide a file or specify all material indices")

    return args


def generate_async(args, ads_ind, bulk_ind, surface_ind):
    # args, ads_ind, bulk_ind, surface_ind = args_and_indices
    job = StructureGenerator(
        args,
        bulk_index=int(bulk_ind),
        surface_index=int(surface_ind),
        adsorbate_index=int(ads_ind),
    )
    job.run()


if __name__ == "__main__":
    """
    This script allows you to either pass in the bulk/surface/adsorbate indices,
    or read from a file containing multiple sets of indices, which you can also
    break up into chunks for parallelizaiton.
    """
    args = parse_args()

    if args.indices_file:
        with open(args.indices_file, "r") as f:
            all_indices = f.readlines()
        inds_to_run = range(len(all_indices))
        if args.chunk_index is not None:
            inds_to_run = range(
                args.chunk_index * args.chunk_size,
                min((args.chunk_index + 1) * args.chunk_size, len(all_indices)),
            )
        print("Running lines", inds_to_run)

        pool_inputs = []
        pool = mp.Pool(args.workers)
        for index in inds_to_run:
            ads_ind, bulk_ind, surface_ind = all_indices[index].strip().split("_")
            pool.apply_async(
                generate_async, args=(args, ads_ind, bulk_ind, surface_ind)
            )

        pool.close()
        pool.join()

    else:
        job = StructureGenerator(
            args,
            bulk_index=args.bulk_index,
            surface_index=args.surface_index,
            adsorbate_index=args.adsorbate_index,
        )
        job.run()

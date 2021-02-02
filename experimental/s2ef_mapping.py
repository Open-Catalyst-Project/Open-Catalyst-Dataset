from tqdm import tqdm
import glob
import ase.io
import pickle
import os
import multiprocessing as mp
import numpy as np


def get_mapping(system):
    randomid = os.path.basename(system)[:-7]
    old_images = ase.io.read(system, ":")
    old_energies = [image.get_potential_energy() for image in old_images]

    new_images = ase.io.read(f"/home/jovyan/projects/ocp/data/trajs/train_02_01/{randomid}.traj", ":")
    new_energies = [image.get_potential_energy() for image in new_images]

    dict_mapping = {}
    # maps old frame idx -> new frame idx
    for old_idx, energy in enumerate(old_energies):
        new_idx = np.where(new_energies == energy)[0]
        try:
            assert len(new_idx) == 1 # ensure no duplicates
        except:
            return (randomid, False)

        dict_mapping[old_idx] = new_idx.item()
    return (randomid, dict_mapping)

if __name__ == "__main__":
    old_systems = glob.glob("/home/jovyan/projects/ocp/data/trajs/train/*.extxyz")[:10]

    pool = mp.Pool(30)
    results = list(tqdm(pool.imap(get_mapping, old_systems), total=len(old_systems)))

    issues = []
    system_mappings = {}

    for system, output in results:
        if output:
            system_mappings[system] = output
        else:
            issues.append(system)

    # write issues and system_mappings to disk

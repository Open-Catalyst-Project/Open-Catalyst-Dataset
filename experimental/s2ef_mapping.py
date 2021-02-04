import sys
import glob
import ase.io
import pickle
import os
import multiprocessing as mp
from tqdm import tqdm
import numpy as np


def read_old(randomid):
    dirs = [
        "trajectories", "trajectories_phase2", "trajectories_val_test"
    ]
    for folder in dirs:
        path = "/global/cfs/cdirs/m2755/OC20_dataset/{}/{}.traj".format(
            folder, randomid
        )
        if os.path.isfile(path):
            return ase.io.read(path, ":")

def read_new(randomid):
    dirs = [
        "train", "test", "val"
    ]
    for folder in dirs:
        path = "/global/cfs/cdirs/m2755/OC20_dataset/new_trajs_02_03/{}/{}.traj".format(
            folder, randomid
        )
        if os.path.isfile(path):
            return ase.io.read(path, ":")

def get_mapping(randomid):
    try:
        old_images = read_old(randomid)
        old_pos = [image.get_positions() for image in old_images]

        new_images = read_new(randomid)
        new_pos = [image.get_positions() for image in new_images]

        dict_mapping = {}
        # maps old frame idx -> new frame idx
        for old_idx, pos in enumerate(old_pos):
            for new_idx, pos_2 in enumerate(new_pos):
                if (pos == pos_2).all():
                    dict_mapping[old_idx] = new_idx
                    break
        assert len(set(dict_mapping.keys())) == len(set(dict_mapping.values()))
    except Exception:
        return (randomid, False)

    return (randomid, dict_mapping)


if __name__ == "__main__":
    all_frames = open(
        "/global/cfs/cdirs/m2755/OC20_dataset/splits/ef_frames.txt", "r"
    ).read().splitlines()

    chunk = int(sys.argv[1])
    slurm_chunks = np.array_split(all_frames, 20)
    ef_frames = slurm_chunks[chunk]

    pool = mp.Pool(mp.cpu_count())
    results = list(
        tqdm(pool.imap(get_mapping, ef_frames), total=len(ef_frames))
    )

    issues = []
    system_mappings = {}

    for system, output in results:
        if output:
            system_mappings[system] = output
        else:
            issues.append(system)

    with open(f"ef_mappings/issues_ef_mapping_chunk_{chunk}.txt", "a") as f:
        for system in issues:
            f.write(f"{system}\n")

    with open(f"ef_mappings/ef_mappings_chunk_{chunk}.pkl", "wb") as f:
        pickle.dump(system_mappings, f)

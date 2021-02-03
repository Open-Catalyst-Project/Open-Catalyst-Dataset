from tqdm import tqdm
import glob
import ase.io
import pickle
import os
import multiprocessing as mp
import numpy as np


def get_mapping(system):
    randomid = os.path.basename(system)[:-7]
    try:
        old_images = ase.io.read(system, ":")
        old_pos = [image.get_positions() for image in old_images]

        new_images = ase.io.read(f"/home/jovyan/projects/ocp/data/trajs/train_02_01/{randomid}.traj", ":")
        new_pos = [image.get_positions() for image in new_images]

        dict_mapping = {}
        # maps old frame idx -> new frame idx
        for old_idx, pos in enumerate(old_pos):
            for new_idx, pos_2 in enumerate(new_pos):
                if (pos == pos_2).all():
                    dict_mapping[old_idx] = new_idx
                    break
    except:
        return (randomid, False)

    return (randomid, dict_mapping)

if __name__ == "__main__":
    old_systems = glob.glob("/home/jovyan/projects/ocp/data/trajs/train/*.extxyz")

    pool = mp.Pool(30)
    results = list(tqdm(pool.imap(get_mapping, old_systems), total=len(old_systems)))

    issues = []
    system_mappings = {}

    for system, output in results:
        if output:
            system_mappings[system] = output
        else:
            issues.append(system)

    with open("s2ef_mapping_issues_pos.txt", "a") as f:
        for system in issues:
            f.write(f"{system}\n")

    with open("s2ef_mappings_pos.pkl", "wb") as f:
        pickle.dump(system_mappings, f)

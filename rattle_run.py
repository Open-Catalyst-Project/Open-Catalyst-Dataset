import numpy as np 
import pickle
import sys
import os
import ase.io
from ocdata.vasp import run_vasp, write_vasp_input_files, VASP_FLAGS

def get_traj_file(pklfile, index):
    with open(pklfile, 'rb') as f:
        ans = pickle.load(f)
        return ans[int(index)]

def get_system_id(trajname):
    return int(trajname.split("/")[-1].split(".")[0][6:])


def main():

    std = float(sys.argv[1]) # 
    index = sys.argv[2]
    metafname = "extracted_files_june16_2020.pkl"

    traj_name = "/checkpoint/sidgoyal/electro_done/random1189878.traj" # get_traj_file(metafname, index)
    opdir = "/checkpoint/sidgoyal/electro_adbulk_rattled_data_round1_june16_2020/"

    full_traj = ase.io.read(traj_name, ":")
    sys_id = get_system_id(traj_name)

    np.random.seed(sys_id) # seed the sampler 
    n = len(full_traj)
    val = int(n * 0.2)
    assert val > 0

    # sample 20% of frames in the trajectory
    frame_ids = np.random.choice(n, val, replace=False)

    # set number of outer loop iterations to 1 
    VASP_FLAGS['nsw'] = 1

    for frame in frame_ids:
        image = full_traj[frame]
        rattled_image = image.copy()
        rattled_image.rattle(stdev=std, seed=100) # fixed seed for all inputs

        current_opdir =  opdir + "/random" + str(sys_id) + "_" + "frame" + str(frame)
        write_vasp_input_files(rattled_image, current_opdir, VASP_FLAGS)


if __name__ == "__main__":
    main()


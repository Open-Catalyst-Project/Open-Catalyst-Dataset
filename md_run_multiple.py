import numpy as np 
import pickle
import sys
import os
import ase.io
from ocdata.vasp import run_vasp, write_vasp_input_files, VASP_FLAGS

def get_traj_files(pklfile, index, count):
    with open(pklfile, 'rb') as f:
        ans = pickle.load(f)
        return ans[index:(index+count)]


def get_system_id(trajname):
    return int(trajname.split("/")[-1].split(".")[0][6:])


def process_for_md(traj_name, opdir, index):

    full_traj = ase.io.read(traj_name, ":")
    sys_id = get_system_id(traj_name)

    np.random.seed(sys_id) # seed the sampler 
    n = len(full_traj)
    assert n > 0


    # set number of outer loop iterations to 40
    VASP_FLAGS['nsw'] = 40

    # modify IBRION to 0 for MD
    VASP_FLAGS['ibrion'] = 0

    VASP_FLAGS['laechg'] = False # don't save CHG

    # adding the following from Muhammed's INCAR file
    VASP_FLAGS['potim'] = 2
    VASP_FLAGS['tebeg'] = 900 # 900 K
    VASP_FLAGS['teend'] = 900 # 900 K
    VASP_FLAGS['mdalgo'] = 0 
    VASP_FLAGS['smass'] =  -3

    last_image = full_traj[-1].copy()
    current_opdir =  opdir + "/randommd" + str(sys_id)

    write_vasp_input_files(last_image, current_opdir, VASP_FLAGS)

    print("Done", index, sys_id, opdir)


def main():

    start_index = sys.argv[1]
    count = sys.argv[2]

    opdir = "/checkpoint/sidgoyal/electro_md_june23/round1_728k/" 

    from pathlib import Path
    Path(opdir).mkdir(parents=True, exist_ok=True)

    metafname = "/private/home/sidgoyal/Open-Catalyst-Dataset/rattle_data_construct_pickle/june16_2020_flist.pkl"

    traj_names = get_traj_files(metafname, int(start_index), int(count))
    for i,t in enumerate(traj_names):
        try:
            process_for_md(t, opdir, int(start_index)+i )
        except:
            print("Failed",  int(start_index)+i, get_system_id(t))

if __name__ == "__main__":
    main()


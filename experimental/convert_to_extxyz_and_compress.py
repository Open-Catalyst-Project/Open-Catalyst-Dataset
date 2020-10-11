import numpy as np
import pickle
import pathlib
import ase
import ase.io
import multiprocessing
import sys
import ase.io.extxyz as extxyz
from from_zack import write_xyz

def write_xyz_file(images, fname):
    columns = (['symbols','positions', 'move_mask', 'tags'])
    with open(fname,'w') as f:
        write_xyz(f, images, columns=columns)

def write_index_file(fname, ans):
    with open(fname, 'w') as f:
        for i in ans:
            f.write(i + "\n")

def write_lzma(inpfile, outfile):
    with open(inpfile, 'rb') as f:
        contents = lzma.compress(f.read(), preset=9)
        with open(outfile, 'wb') as op:
            op.write(contents)

def combine_and_add_tag_to_images(ef_chunk, tag_map, ene_map):
    #/checkpoint/electrocatalysis/relaxations/bulkadsorbate/chunk43/random2101429.traj,115,200,random2101429

    images = []
    frame_info = []
    for trajname, str_frame_index, total_frames, adbulk_id in ef_chunk:
        frame_index = int(str_frame_index)
        current_image = ase.io.read(trajname, index=frame_index)
        tags = np.array(tag_map[adbulk_id])
        current_image.set_tags(tags)
        images.append(current_image) 
        frame_info.append(adbulk_id + ",frame" + str_frame_index + "," + str(ene_map[adbulk_id]))
    return images, frame_info

def read_traj_write_xyz_lzma(indices, ef_train_chunks, tag_map, ene_map, opdir):

    for index in indices:
        ef_chunk = ef_train_chunks[index]
        
        output_prefix = opdir + "/" + str(index)
        output_traj_fname = output_prefix + ".extxyz"
        output_index_fname = output_prefix + ".txt"

        # extract relevant information
        tagged_images, frame_info = combine_and_add_tag_to_images(ef_chunk, tag_map, ene_map)

        # write xyz and data files
        write_xyz_file(tagged_images, output_traj_fname)
        write_index_file(output_index_fname, frame_info)
        
        # write lzma compressed versions of xyz and data files
        write_lzma(output_traj_fname, output_traj_fname + ".xz")
        write_lzma(output_index_fname, output_index_name + ".xz")


def read_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def read_file(fname):
    lst = []
    with open(fname, 'r') as f:
        for line in f:
            temp = line.rstrip()
            val = temp.split(",")
            lst.append(val)
    return lst

def divide_work(ef_train_frames, n):
    return [ef_train_frames[i:i + n] for i in range(0, len(ef_train_frames), n)]

if __name__ == "__main__":

    #EF task
    # format of split files is 001.txt upto 134.txt
    split_name = sys.argv[1].zfill(3) + ".txt"
    ef_train_fname = "/checkpoint/sidgoyal/electro_downloadables/ef_master_file_splits/" + split_name 

    ef_train_frames = read_file(ef_train_fname)

    # pkl with tag info
    tag_file = "/checkpoint/electrocatalysis/relaxations/mapping_old/pickled_mapping/adslab_tags_full.pkl"
    tag_map = read_pkl(tag_file)

    opdir = "/checkpoint/sidgoyal/electro_downloadables/ef2M_ALL/train/"
    p = pathlib.Path(opdir)
    p.mkdir(parents=True, exist_ok=True)

    ene_file = "/checkpoint/electrocatalysis/relaxations/mapping/new_adslab_ref_energies_09_22_20.pkl"
    ene_map = read_pkl(ene_file)

    # each chunk will have 5000 images
    frames_per_traj = 5000

    ef_train_chunks = divide_work(ef_train_frames, frames_per_traj)
   
    # parallel processing
    manager = multiprocessing.Manager()
    ans = manager.list()

    k = 41 # we are requesting 41 cores from slurm 

    print(len(ef_train_chunks))
    indices = [i for i in range(len(ef_train_chunks))]
    tasks = [indices[i::k] for i in range(k)]
    procs = []
    # instantiating processes
    for t in tasks:
        proc = multiprocessing.Process(target=read_traj_write_xyz_lzma, args=(t, ef_train_chunks, tag_map, ene_map, opdir)) 
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()

    print("Done")

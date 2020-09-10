import ase
import ase.io
import glob
import numpy as np
import json
import gzip
import bz2
import lzma
import pickle
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from monty.serialization import dumpfn
import os
import shutil
import copy
import ase.io.extxyz as extxyz

def read_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def vanilla_map(adbulk_id, energies, forces, positions, elements, tags, cell, adenergy):
    d = {}
    d['id'] = adbulk_id
    d['energy'] = energies
    d['force'] = forces
    d['pos'] = positions
    d['tag'] = tags
    d['cell'] = cell
    d['elements'] = elements
    d['adsor_ene'] = adenergy
    return d

def optimized_map(adbulk_id, energies, forces, positions, elements, tags, cell, adenergy):
    d = {}

    cell = np.array(cell, dtype=np.float64).reshape((3, 3))
    inv_cell = np.linalg.inv(cell)

    fractional_coords = []
    for pos in positions:
        frac = np.dot(np.array(pos), inv_cell)
        fractional_coords.append(frac.tolist())


    d['id'] = adbulk_id
    d['energy'] = energies
    d['force'] = forces
    d['frac_coor'] = fractional_coords
    d['tag'] = tags
    d['cell'] = cell.tolist()
    d['elements'] = elements
    d['adsor_ene'] = adenergy

    
    # check if values read from the dictionary lead to original cartesian coordinates
    cell_read = np.array(d['cell'], dtype=np.float64).reshape((3,3))
    check_cart_coords = []
    for i in range(len(d['frac_coor'])):
        frac_np = np.array(d['frac_coor'][i]) 
        current = np.dot(frac_np, cell_read)
        assert np.allclose(current, np.array(positions[i]))

    return d


def pymatgen_map(adbulk_id, fname, energies, tags, adenergy):
    images =  ase.io.read(fname, ":") #ase.io.trajectory.Trajectory(fname)
    structs = [AseAtomsAdaptor.get_structure(atoms) for atoms in images]
    for i in range(len(structs)):
        struct = structs[i]
        image = images[i]
        struct.add_site_property('forces', image.get_forces(apply_constraint=False))
                 
    pymatgen_traj = Trajectory.from_structures(structs)
    pymatgen_traj.frame_properties = {"energy": energies, 
                                      "tags": [tags]*len(structs),  
                                      "adsor_ene": [adenergy]*len(structs), 
                                      "id": [adbulk_id]*len(structs)}
    # we are duplicating this as "each property should have a length equal to the trajectory length. "
    return pymatgen_traj



def get_json(d):
    json_str = json.dumps(d)
    json_bytes = json_str.encode('utf-8')
    return json_bytes

def write_specific(d, outfile, ext, binarized=False):
    if not binarized:
        d = get_json(d)

    if ext == "gz":
        contents = gzip.compress(d, 9)

    elif ext == "bz2":
        contents = bz2.compress(d, 9)

    elif ext == "xz":
        contents = lzma.compress(d, preset=9)
        
    with open(outfile, 'wb') as op:
        op.write(contents)

def xyz_map(adbulk_id, fname, adenergy, tags, cell):
#    all_images = ase.io.trajectory.Trajectory(fname)
    all_images = ase.io.read(fname, ":")
    tags = np.array(tags)
    for image in all_images:
        image.set_tags(tags)

    return all_images

def write_xyz(images, fname):
    columns = (['symbols','positions', 'move_mask', 'tags'])
    with open(fname,'w') as f:
        extxyz.write_xyz(f, images, columns=columns)


                
def experiment_various_formats(fname, tag_map, adsorption_energy_map, temp_dir):

    images = ase.io.read(fname, ":") #ase.io.trajectory.Trajectory(fname)
    energies = [image.get_potential_energy(apply_constraint=False) for image in images]
    forces = [image.get_forces(apply_constraint=False).tolist() for image in images]
    positions = [image.get_positions().tolist() for image in images]
    elements = images[0].get_atomic_numbers().tolist()
    cell = np.array(images[0].get_cell(complete=True)).tolist() #complete is to get a 3x3
    # https://github.com/rosswhitfield/ase/blob/dee5e78f8df4016a9c65c4599331001294e5e2d6/ase/geometry/cell.py#L202


    adbulk_id = fname.split("/")[-1].split(".")[0]
    if adbulk_id not in adsorption_energy_map:
        return {}
    adenergy = adsorption_energy_map[adbulk_id]
    tags = tag_map[adbulk_id].tolist()

    ans = {}

    # 1. vanilla format with everything
    vanilla = vanilla_map(adbulk_id, energies, forces, positions, elements, tags, cell, adenergy)
    
    # 2. optimized format with fractional coordinates
    optimized = optimized_map(adbulk_id, energies, forces, positions, elements, tags, cell, adenergy)

    # 3. pymatgen
    pymat = pymatgen_map(adbulk_id, fname, energies, tags, adenergy)

    # 4. extxyz format
    xyz = xyz_map(adbulk_id, fname, adenergy, tags, cell)
           
    xyz_filename = temp_dir + adbulk_id + "_xyz.extxyz"
    write_xyz(xyz, xyz_filename)

    # different formats:
    # gz / bz2 / lzma
    
    for ext in ['gz', 'bz2', 'xz']:
        type1 = 'vanilla_json_' + ext
        type1_fname = temp_dir + adbulk_id + '_vanilla.json.' + ext
        write_specific(vanilla, type1_fname, ext)
        ans[type1] = os.path.getsize(type1_fname)

        type2 = 'optimized_json_' + ext
        type2_fname = temp_dir + adbulk_id + '_optimized.json.' + ext
        write_specific(optimized, type2_fname, ext)
        ans[type2] = os.path.getsize(type2_fname)
        
        # for pymatgen
        if ext != 'xz':
            type3 = 'pymat_json_' + ext
            type3_fname = temp_dir + adbulk_id + '_pymatgen.json.' + ext
            dumpfn(pymat, type3_fname)
            ans[type3] = os.path.getsize(type3_fname)

            type4 = 'optimized_monty_json_' + ext
            type4_fname = temp_dir + adbulk_id + '_optimized_monty.json.' + ext
            dumpfn(optimized, type4_fname)
            ans[type4] = os.path.getsize(type4_fname)

        # for xyz
        xyz_file_content = open(xyz_filename, 'rb').read()
        type5 = "exyz_" + ext
        type5_fname = temp_dir + adbulk_id + "_xyz.extxyz." + ext
        write_specific(xyz_file_content, type5_fname, ext, binarized=True)
        ans[type5] = os.path.getsize(type5_fname)


    ans['extxyz'] = os.path.getsize(xyz_filename)
    ans['regular'] = os.path.getsize(fname) #.traj file


    regular_gz_name = temp_dir + adbulk_id + '.traj.gz'
    with open(fname, 'rb') as f_in:
        with gzip.open(regular_gz_name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    ans['regular_gz'] = os.path.getsize(regular_gz_name) #.traj.gz file


    # checking pkl version
    pkl_optimized_name = temp_dir + adbulk_id + ".optimized_pkl.bz2"
    pickle.dump(optimized, bz2.open(pkl_optimized_name, 'wb'))
    ans['optimized_pkl_bz2'] = os.path.getsize(pkl_optimized_name)

    # normalize size values
    base = ans['regular']
    for key in ans:
        ans[key] = base / ans[key]

    return ans

def main():
    filedir = "traj_files/"
    filelist = glob.glob(filedir + "*.traj")

    tag_map = read_pkl('/checkpoint/electrocatalysis/relaxations/mapping/pickled_mapping/adslab_tags_full.pkl')
    adsorption_energy_map = read_pkl('/checkpoint/electrocatalysis/relaxations/mapping/pickled_mapping/adsorption_energies_full.pkl')

    temp_dir = "temp/"

    mega_ans = {}
    for i, fname in enumerate(filelist):
        ans = experiment_various_formats(fname, tag_map, adsorption_energy_map, temp_dir)

        # ans will be a dict with: storage_format -> memory_saving
        for key in ans:
            if key not in mega_ans:
                mega_ans[key] = [ans[key]]
            else:
                mega_ans[key].append(ans[key])

        print(i)
    # dictionary with avg memory speedup values
    speedup = {}
    for i, key in enumerate(mega_ans):
        lst = np.array(mega_ans[key])
        speedup[key] = np.mean(lst)
    sitems = sorted(speedup.items(), key=lambda x: x[1])
    for tup in sitems:
        print(tup[0], tup[1])



if __name__ == "__main__":
    main()

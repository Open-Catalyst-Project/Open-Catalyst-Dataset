import pickle
import ase


def find_unique(dir_prefix, index_list, system_type):
    
    d = {} # going from sampling-tree-information to number-of-atoms

    key_atomobj = "adsorbed_bulk_atomsobject"
    key_ssample = "adsorbed_bulk_samplingstr"
    if system_type == "bulk":
        key_atomobj = "bulk_atomsobject"
        key_ssample = "bulk_samplingstr"

    for i in index_list:
        current_sys_path = dir_prefix + "random" + str(i)  
        metadata_path = current_sys_path + "/metadata.pkl"
        with open(metadata_path, 'rb') as f:
            struct_dict = pickle.load(f)
            atomobj = struct_dict[key_atomobj]
            ssample = struct_dict[key_ssample]
            
            if ssample not in d:
                d[ssample] = (current_sys_path, len(atomobj))
            else:
                print(ssample, current_sys_path, d[ssample])

    print(system_type, len(index_list), len(d))
    return d.values()


def read_generated_systems(fname):
    lst = []
    with open(fname, 'r') as f:
        for line in f:
            lst.append(line.rstrip())
    return lst

def write_pairs(ans_list, fname):
    with open(fname, 'w') as f:
        for tup in ans_list:
            f.write(tup[0] + " " + str(tup[1]) + "\n")


def main():
    index_list = read_generated_systems('successful_systems_may3_round1.txt')

    BULK_DIR_PREFIX = "/checkpoint/sidgoyal/electro_may3_100k/bulk/"
    ADSORBED_BULK_DIR_PREFIX = "/checkpoint/sidgoyal/electro_may3_100k/bulkadsorbate/"

    bulk_ans_list = find_unique(BULK_DIR_PREFIX, index_list, "bulk")
    adsorbed_bulk_ans_list = find_unique(ADSORBED_BULK_DIR_PREFIX, index_list, "adsbulk") 

    bulk_opname = "unique_bulk_systems_may3_round1.txt"
    adsorbed_bulk_opname = "unique_adsorbed_bulk_systems_may3_round1.txt"

    write_pairs(bulk_ans_list, bulk_opname)
    write_pairs(adsorbed_bulk_ans_list, adsorbed_bulk_opname)


if __name__ == "__main__":
    main()

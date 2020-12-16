from ocdata.adsorptions import sample_structures
from ocdata.vasp import run_vasp, write_vasp_input_files
import numpy as np
import time
import sys
import pickle

val = sys.argv[1]

np.random.seed(int(val))

BULK_PATH = "/checkpoint/sidgoyal/electro_may6_100k_round2/bulk/"
ADSORBED_BULK_PATH = "/checkpoint/sidgoyal/electro_may6_100k_round2/bulkadsorbate/"

def write_pkl(d, path):
    with open(path, 'wb') as f:
        pickle.dump(d, f)

s = time.time()
success = True
try:
    adsorbed_bulk_dict, bulk_dict = sample_structures()  

    adsorbed_bulk_dir = ADSORBED_BULK_PATH + "/random" + val + "/"
    bulk_dir = BULK_PATH + "/random" + val + "/"

    write_vasp_input_files(adsorbed_bulk_dict["adsorbed_bulk_atomsobject"], adsorbed_bulk_dir)
    write_vasp_input_files(bulk_dict["bulk_atomsobject"], bulk_dir)

    write_pkl(bulk_dict, bulk_dir + "metadata.pkl")
    write_pkl(adsorbed_bulk_dict, adsorbed_bulk_dir + "metadata.pkl")
    
except:
    success = False
e = time.time()

out = "Error"
if success:
    out = "Done"

print("Seed:", val, "time:", round(e - s, 2), out)



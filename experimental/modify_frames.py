import os
import pickle
from tqdm import tqdm
import glob


paths = glob.glob("/home/jovyan/projects/aws-backup/splits_01_18/updated_splits/EF/*.txt")
mapping = pickle.load(open("/home/jovyan/projects/ocp/data/trajs/ef_mappings/all_s2ef_traj_maps.pkl", "rb"))
dirmapping = {
    "train":"/checkpoint/electrocatalysis/relaxations/bulkadsorbate/restitch_jan2021",
    "val":"/checkpoint/electrocatalysis/relaxations/bulkadsorbate/restitch_jan2021_validation_all",
    "test":"/checkpoint/electrocatalysis/relaxations/bulkadsorbate/restitch_jan2021_test_all"
}

for path in tqdm(paths):
    split = os.path.basename(path).split("_")[0]
    issues = 0
    with open(
        os.path.join("02_04_updated_ef_splits", os.path.basename(path)), "a"
    ) as f:
        for line in tqdm(open(path)):
            try:
                olddir, frame, total, randomid = line.split()[0].split(",")
                newframe = str(mapping[randomid][int(frame)])
                newdir = os.path.join(dirmapping[split], randomid+".traj")
                new_line = ",".join([newdir,newframe,total, randomid])+"\n"
                f.write(new_line)
            except:
                issues += 1
    print(path)
    print(issues)

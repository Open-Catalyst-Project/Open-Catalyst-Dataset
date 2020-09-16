import os
import sys
import pickle
import subprocess

def main():
    fname = "./samples_to_run.pkl"
    f = open(fname, 'rb')
    lst = pickle.load(f)

    id = sys.argv[1]

    tgt_path = lst[int(id)]

    if os.path.isdir(tgt_path):
        os.chdir(tgt_path)

        print("Started", tgt_path)
        cmd = ["/private/home/sidgoyal/intel/impi/5.0.2.044/intel64/bin/mpirun", "-np",  "36",  "/private/home/sidgoyal/vasp5.4.4/Linux-x86_64/vasp_parallel"]
        f = open("stdout.log", 'w')
        g = open("stderr.log", 'w')
        p = subprocess.Popen(cmd, stdout=f, stderr=g)
        p.wait()
        f.close()
        g.close()

        print("Done", tgt_path)
    else:
        print(f"{tgt_path} does not exist!")


if __name__ == "__main__":
    main()

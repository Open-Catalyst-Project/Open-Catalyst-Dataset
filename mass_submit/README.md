1. Run `create_path_files.py` to create a file containing paths to all vasp jobs to be submitted. To be run in the directory containing all the subdirectories with vasp input files.
2. Modify the `submit.sh` script to update the length of the job array (total jobs to be submitted based off the above step).

#!/bin/bash

## job name
#SBATCH --job-name=rattle
#SBATCH --output=slurm_logs_06_13_20/%A_%a.out
#SBATCH --error=slurm_logs_06_13_20/%A_%a.err

#SBATCH --partition=priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=41
#SBATCH --mem-per-cpu=6g
#SBATCH --time=360
#SBATCH --constraint=pascal
#SBATCH --comment="non-preemtable-cpu-job"
#SBATCH --array=0-6000%100

export LD_LIBRARY_PATH=/private/home/sidgoyal/intel/impi/5.0.2.044/intel64/lib/:${LD_LIBRARY_PATH}

start=${SLURM_ARRAY_TASK_ID}
BEGIN=0
let start=BEGIN+SLURM_ARRAY_TASK_ID*1
let potend=start+1
endlimit=6000

end=$(( potend < endlimit ? potend : endlimit ))

for (( i=${start}; i < ${end}; i++ )); do
   python run_vasp_with_id.py $i  & 
done
wait


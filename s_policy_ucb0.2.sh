#!/bin/bash

#SBATCH -q normal
#SBATCH -N 4
#SBATCH -t 08:00:00
#SBATCH -J  policy_test_eps
#SBATCH --mail-user=zalbasri@stanford.edu
#SBATCH --mail-type=ALL


module purge
module load openmpi
module load anaconda3 
module load julia 
eval "$(conda shell.bash hook)"
conda activate cs238env
cd /home/zalbasri/CS238/CS238_Final
python3 framework_ucb02.py 

#!/bin/bash

#SBATCH -q normal
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH -J  epsilon_decay0597_test
#SBATCH --mail-user=zalbasri@stanford.edu
#SBATCH --mail-type=ALL


module purge
module load openmpi
module load anaconda3 
module load julia 
eval "$(conda shell.bash hook)"
conda activate cs238env
cd /home/zalbasri/CS238/CS238_Final
python3 epsilon_decay0597_test.py 

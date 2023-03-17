#!/bin/bash

#SBATCH -q normal
#SBATCH -N 4
#SBATCH -t 08:00:00
#SBATCH -J  policy_real
#SBATCH --mail-user=nkozak@stanford.edu
#SBATCH --mail-type=ALL


module purge
module load openmpi
module load anaconda3 
module load julia 
conda activate env_CS238
cd /home/nkozak/CS238/frame_testing_cont
python3 policy_real.py 

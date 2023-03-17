#!/bin/bash

#SBATCH -q normal
#SBATCH -N 4
#SBATCH -t 08:00:00
#SBATCH -J  softmax_1_pt1
#SBATCH --mail-user=stoccop@stanford.edu
#SBATCH --mail-type=ALL

module purge
module load openmpi
module load anaconda3 
module load julia 
eval "$(conda shell.bash hook)"
conda activate env_CS238
cd /home/stoccop/AA228 #/home/nkozak/CS238/frame_testing_cont
python3 sm_1_pt1.py

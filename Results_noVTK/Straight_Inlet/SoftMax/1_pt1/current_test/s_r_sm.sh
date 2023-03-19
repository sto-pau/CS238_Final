#!/bin/bash

#SBATCH --nodes=1
#SBATCH -q normal
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH -J  sme
#SBATCH --mail-user=stoccop@stanford.edu
#SBATCH --mail-type=ALL

module purge
module load openmpi
module load anaconda3 
module load julia 
eval "$(conda shell.bash hook)"
conda activate env_CS238
cd /home/stoccop/AA228/softmax_1_0pt1_run #/home/nkozak/CS238/frame_testing_cont
python3 sm_1_pt1.py

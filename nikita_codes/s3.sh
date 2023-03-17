#!/bin/bash

#SBATCH -q normal
#SBATCH -N 4
#SBATCH -t 08:00:00
#SBATCH -J  s3
#SBATCH --mail-user=nkozak@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --output=s3.%j.out

module purge
module load openmpi
module load anaconda3 
module load julia 
eval "$(conda shell.bash hook)"
conda activate env_CS238
cd /home/nkozak/CS238/explore_states
python3 s3.py

#reduce log
tail -n 1000 s3*.out > end_s3.out
rm s3*.out

#remove files
cd /home/nkozak/CS238/explore_states/s3/backGround
rm -r 1*
rm -r 2*
rm -r 3*
rm -r 4*
rm -r 5*
rm -r 6*
rm -r 7*
rm -r 8*
rm -r 9*
rm -r processor*

#reduce storage
zip -r VTK.zip VTK
rm -r VTK
zip -r postProcessing.zip postProcessing
rm -r postProcessing

#move to scratch
mkdir /scratch/users/nkozak/CS238/explore_states/s3
mv /home/nkozak/CS238/explore_states/s3/backGround /scratch/users/nkozak/CS238/explore_states/s3/.

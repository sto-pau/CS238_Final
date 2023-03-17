#!/bin/bash

#SBATCH -q normal
#SBATCH -N 4
#SBATCH -t 06:00:00
#SBATCH -J TEST_1
#SBATCH --mail-user=nkozak@stanford.edu
#SBATCH --mail-type=ALL


module purge
module load openmpi
cd /home/nkozak/CS238/test_run1/backGround
decomposePar
mpirun -np 4 renumberMesh -overwrite -parallel 
mpirun -np 4 overPimpleDyMFoam -parallel 
reconstructPar -newTimes
foamToVTK

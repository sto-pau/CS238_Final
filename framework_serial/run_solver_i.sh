#!/bin/bash
cd backGround
module load openmpi
#decomposePar
#mpirun -np 4 renumberMesh -overwrite -parallel
#mpirun -np 4 overPimpleDyMFoam -parallel
renumberMesh -overwrite
overPimpleDyMFoam
#reconstructPar -latestTime#-newTimes
foamToVTK -latestTime
cd ..

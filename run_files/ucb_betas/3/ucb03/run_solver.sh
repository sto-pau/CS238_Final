#!/bin/bash
cd backGround
module load openmpi
#mpirun -np 4 overPimpleDyMFoam -parallel
overPimpleDyMFoam
#reconstructPar -latestTime#-newTimes
foamToVTK -latestTime
cd ..

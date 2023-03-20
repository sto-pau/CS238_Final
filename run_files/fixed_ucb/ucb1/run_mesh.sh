#!/bin/bash

module purge
module load openmpi

foamCleanTutorials

cd airfoil
blockMesh
snappyHexMesh -overwrite | tee log.snappyHexMesh
extrudeMesh
createPatch -overwrite
transformPoints -translate '(0 -0.5 0)'
cd ..

cd backGround
blockMesh
mergeMeshes . ../airfoil -overwrite
topoSet
topoSet -dict system/topoSetDict_movingZone

rm -r 0
cp -r 0_org 0

checkMesh |  tee log.checkMesh
setFields | tee log.setFields

cd ..

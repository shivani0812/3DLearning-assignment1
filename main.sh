#!/bin/bash
mkdir -p output

echo "############################################"
echo "Running question 1.1 360-degree Renders"
echo "############################################"
python3 main.py -q 1.1
echo '------------DONE------------'

echo "############################################"
echo "Running questiion 1.2 Re-creating the Dolly Zoom"
echo "############################################"
python3 main.py -q 1.2
echo '------------DONE------------'

echo "############################################"
echo "Running question 2.1 Constructing a Tetrahedron"
echo "############################################"
python3 main.py -q 2.1
echo '------------DONE------------'

echo "############################################"
echo "Running question 2.2 Constructing a Cube"
echo "############################################"
python3 main.py -q 2.2
echo '------------DONE------------'

echo "############################################"
echo "Running question 3 Re-texturing a mesh"
echo "############################################"
python3 main.py -q 3
echo '------------DONE------------'

echo "############################################"
echo "Running question 4 Camera Transformations"
echo "############################################"
python3 main.py -q 4
echo '------------DONE------------'

echo "############################################"
echo "Running question 5.1 Rendering Point Clouds from RGB-D Images"
echo "############################################"
python3 main.py -q 5.1
echo '------------DONE------------'

echo "############################################"
echo "Running question 5.2 Parametric Functions"
echo "############################################"
python3 main.py -q 5.2
echo '------------DONE------------'

echo "############################################"
echo "Running question 5.3 Implicit Surfaces"
echo "############################################"
python3 main.py -q 5.3
echo '------------DONE------------'

echo "############################################"
echo "Running question 6 Do Something Fun"
echo "############################################"
python3 main.py -q 6
echo '------------DONE------------'

echo "############################################"
echo "Running question 7 (Extra Credit) Sampling Points on Meshes "
echo "############################################"
python3 main.py -q 7
echo '------------DONE------------'


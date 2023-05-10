#!/bin/bash
export PYTHONIOENCODING='utf-8'
cd ..

FILE_NAME=$1
export FILE_NAME
sleep 2 
source slurm/${1}



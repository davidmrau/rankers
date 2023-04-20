#!/bin/bash
export PYTHONIOENCODING='utf-8'
cd ..
nvidia-smi

FILE_NAME=$1
export FILE_NAME
sleep 5
source slurm/${1}



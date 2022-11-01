USER=`accinfo | grep NRC | awk '{print $3}'`
#Loading modules
module purge
#module load pre2019
#module load 2019
module load eb
module load 2021
module load CUDA/11.3.1
module load Python/3.9.5-GCCcore-10.3.0
#module load cuDNN
export PYTHONIOENCODING='utf-8'
cd ..

FILE_NAME=$1
export FILE_NAME

source slurm/${1}



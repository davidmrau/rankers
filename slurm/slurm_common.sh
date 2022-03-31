USER=`accinfo | grep NRC | awk '{print $3}'`
#Loading modules
module purge
#module load pre2019
#module load 2019
module load eb
module load cuDNN
#module load Python/3.6.6-foss-2019b
#module load Python/3.7.5-foss-2019b
#module load cuDNN/7.0.5-CUDA-9.0.176
#module load NCCL/2.0.5-CUDA-9.0.176 
export PYTHONIOENCODING='utf-8'
cd ..

FILE_NAME=$1
export FILE_NAME



bash telegram.sh "${USER} ${FILE_NAME} ${SLURM_JOBID} Started"
source slurm/${1}
bash telegram.sh "${USER} ${FILE_NAME} ${SLURM_JOBID} Finished"



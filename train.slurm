#!/usr/bin/env bash
#SBATCH --job-name=chottallm
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --time=14-00:00:00
#SBATCH --output=./logs/chottallm_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

##SBATCH --partition=compute
##SBATCH --array=1-1%1

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "Number of gpus:= " $CUDA_VISIBLE_DEVICES
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "



source /home/kc2819/.bashrc
conda deactivate
conda activate torchdev



if [[ ${SLURM_ARRAY_TASK_ID} == 1 ]] ; then
    # python -m torchrun --nproc_per_node=1 src/runmodel.py
    python src/runmodel.py
fi

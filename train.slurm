#!/usr/bin/env bash
#SBATCH --job-name=chotallm
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --time=14-00:00:00
#SBATCH --output=./logs/chotallm_%A_%a.out
##SBATCH --partition=gpu
##SBATCH --gres=gpu:a100:1

#SBATCH --partition=compute
#SBATCH --array=1-1%1




source /home/kc2819/.bashrc
conda deactivate
conda activate torchdev



if [[ ${SLURM_ARRAY_TASK_ID} == 1 ]] ; then
    python src/runmodel.py
fi

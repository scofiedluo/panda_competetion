#!/bin/bash

#SBATCH --job-name=test_run
#SBATCH --partition=dgx2
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=test_run%j.out
#SBATCH --error=test_run%j.err

ulimit -s unlimited
ulimit -l unlimited

module load miniconda3/4.6.14-gcc-4.8.5
module load gcc/8.3.0-gcc-4.8.5 cuda/10.1.243-gcc-8.3.0

source activate PANDA_tianchi

cd ..
sh run.sh
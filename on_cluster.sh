#!/bin/bash
#SBATCH -A hpc-gpca-lab          
#SBATCH --job-name=fl-compress   
#SBATCH --nodes=3          
#SBATCH --nodelist=dgx-1,dgx-2,dgx-3
#SBATCH --ntasks-per-node=1      
#SBATCH --gres=gpu:1             
#SBATCH --mem=16G                
#SBATCH --time=00:15:00          
#SBATCH --output=logs/job_%j.out

# Make sure logs directory exists
mkdir -p logs

# Add date to log file names
DATE=$(date +%Y%m%d_%H%M%S)

mpirun ./fl-rl-compression-MPI/build/compress c fl-nccl 512mb 512.nccl.c \
  > logs/mpi_512_nccl_${DATE}.log 2>&1

mpirun ./fl-rl-compression-MPI/build/compress c fl-mpi 512mb 512.nccl.c \
  > logs/mpi_512_mpi_${DATE}.log 2>&1

mpirun ./fl-rl-compression-MPI/build/compress c fl-nccl 2048mb 2048.nccl.c \
  > logs/mpi_2048_nccl_${DATE}.log 2>&1

mpirun ./fl-rl-compression-MPI/build/compress c fl-mpi 2048mb 2048.mpi.c \
  > logs/mpi_2048_mpi_${DATE}.log 2>&1

mpirun ./fl-rl-compression-MPI/build/compress c fl-nccl 3124mb 3124.nccl.c \
  > logs/mpi_3124_nccl_${DATE}.log 2>&1

mpirun ./fl-rl-compression-MPI/build/compress c fl-mpi 3124mb 3124.mpi.c \
  > logs/mpi_3124_mpi_${DATE}.log 2>&1
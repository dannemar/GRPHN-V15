#!/bin/bash
#SBATCH --job-name=Graphene_cuda_compile_and_run
#SBATCH --ntasks=2
#SBATCH --partition=a100
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=1-23:59:00
#SBATCH --account=a100free
#SBATCH --output=/scratch/mrsdan024/Graphene/RnD/Code/CUDA/V11_AllACFs/src/JOB_OUT.out
#SBATCH --error=/scratch/mrsdan024/Graphene/RnD/Code/CUDA/V11_AllACFs/src/JOB_ERR.out


module load compilers/gcc-14.3.0

../build/grph_v15_STCFs -w 256 -h 256 -e 100 -s 41 -tt 600 -eqt 500 -dt 0.01 -r 200
../build/grph_v15_STCFs -w 256 -h 256 -e 200 -s 41 -tt 600 -eqt 500 -dt 0.01 -r 200
../build/grph_v15_STCFs -w 256 -h 256 -e 300 -s 41 -tt 600 -eqt 500 -dt 0.01 -r 200
../build/grph_v15_STCFs -w 256 -h 256 -e 400 -s 41 -tt 600 -eqt 500 -dt 0.01 -r 200
../build/grph_v15_STCFs -w 256 -h 256 -e 500 -s 41 -tt 600 -eqt 500 -dt 0.01 -r 200
../build/grph_v15_STCFs -w 256 -h 256 -e 600 -s 41 -tt 600 -eqt 500 -dt 0.01 -r 200
../build/grph_v15_STCFs -w 256 -h 256 -e 700 -s 41 -tt 600 -eqt 500 -dt 0.01 -r 200
../build/grph_v15_STCFs -w 256 -h 256 -e 800 -s 41 -tt 600 -eqt 500 -dt 0.01 -r 200
../build/grph_v15_STCFs -w 256 -h 256 -e 900 -s 41 -tt 600 -eqt 500 -dt 0.01 -r 200
../build/grph_v15_STCFs -w 256 -h 256 -e 1000 -s 41 -tt 600 -eqt 500 -dt 0.01 -r 200

module unload compilers/gcc-14.3.0

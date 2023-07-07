#!/bin/sh

#SBATCH -N 1
#SBATCH -n 12
#SBATCH -t 12:00:00
### SBATCH -o log_py.txt

OMP_NUM_THREADS=12 ../../Downloads/practice/rcbk/build/bin/rcbk -ic MV ${1} ${2} 0.01 ${3} -rc BALITSKY -alphas_scaling ${4} -maxy 12 -fast -output results/bks/${5}.dat

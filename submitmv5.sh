#!/bin/sh

OMP_NUM_THREADS=2 ../../Downloads/practice/rcbk/build/bin/rcbk -ic MV ${1} ${2} 0.01 ${3} -rc BALITSKY -alphas_scaling ${4} -maxy 12 -fast -output results/bks/${5}.dat

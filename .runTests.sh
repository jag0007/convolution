#!/bin/bash

module load blas
module load cuda

echo -ne "\n\nWaiting for job to start...\n\n"

echo -ne "==================\n"
echo -ne "Starting execution\n"
echo -ne "==================\n\n"

# nsys profile build/bin/saxpy_test

# echo -ne "\n\n"

# ncu -k saxpy_kernel -o profile build/bin/saxpy_test

build/bin/conv_test 
#echo -ne "\n\nNext\n\n"
#echo -ne "Running Timer Test\n"
#build/bin/timer_test

echo -ne "\n==================\n"
echo -ne "Finished execution\n"
echo -ne "==================\n\n"
echo "Hit Ctrl + C to exit..."

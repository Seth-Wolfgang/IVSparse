#!/bin/sh

 # https://github.com/ginkgo-project/ssget is required for runBenchmark.sh. 
 # The version of ssget provided is slightly modified to store matrices in 
 # /Benchmarking/matrices.

# echo "Running benchmark with $numCols columns, $numRows rows, $numNonzeros nonzeros, $numMatrices matrices, and \"$problemKind\" problem kind"

# Compiling benchmark
# g++ -g -w -O2 lib/benchmark.cpp -o benchmark -llapack
g++ -w -O2 -I ~/eigen lib/benchmark.cpp -o benchmark -llapack


# Checking if compilation was successful
if [ ! -f "benchmark" ]; then
    exit 1
fi


if [ -f "matrices.txt" ]; then
    rm matrices.txt
fi

# This allows filtering for specific types of matrices
# And cannot be used for attributes such as rows, columns, or nonzeros
if [ $# -ge 1 ]; then
    ./ssget -s '@'$1 > matrices.txt
    echo "Downloading matrices of type $1"
    touch matrices.txt
fi

numMatrices=1000
# Downloading matrices and running benchmark at the same time
for x in $(seq 1 $numMatrices)
do
    # Download matrix
    echo "Downloading matrix $i"

    # If matrices.txt exists, get the ID from it
    # Else we just run ssget in sequential order of IDs
    if [ -f "matrices.txt" ]; then
        # Get the ID of the matrix
        ID=$(head -n 1 matrices.txt)
        # Remove the ID from the file
        sed -i '1d' matrices.txt
        MATRIX_PATH=$(./ssget -t MM -e -i $ID)
    else
        MATRIX_PATH=$(./ssget -t MM -e -i $x)
    fi

    echo "Matrix path: $MATRIX_PATH"
    # Grabs the ID here because its easier to do in shell than C
    id=$(grep -oP '(?<=id: ).*' $MATRIX_PATH)
    echo "id: $id"
    
    echo "Running C++ benchmark for matrix ID: \033[0;32m$id\033[0m"

    
    ./benchmark $MATRIX_PATH $id
    # valgrind -s --leak-check=full --track-origins=yes ./benchmark $MATRIX_PATH $id
    rm -r $MATRIX_PATH
done

# Clean up
rm -rf matrices/
rm benchmark
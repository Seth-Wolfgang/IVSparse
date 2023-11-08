#!/bin/bash
clear
# Step 1: Compile simulatedBench_CSC.cpp with g++ -O2
g++ -O2 simulatedBench_CSC.cpp -o a.out

# Step 2: Iterate through folders in ~/matrices
matrix_dir="/home/sethwolfgang/matrices"

if [ ! -d "$matrix_dir" ]; then
    echo "Error: $matrix_dir directory does not exist."
    exit 1
fi

a=0
for folder in "$matrix_dir"/*; do
    if [ -d "$folder" ]; then
        # echo $folder

        # Step 3: Run simulatedBench_CSC.cpp for each folder
        vals_csv="$folder/vals.csv"
        inner_csv="$folder/inner.csv"
        outer_csv="$folder/outer.csv"
        folder_name="$(basename "$folder")"

        if [ -f "$vals_csv" ] && [ -f "$inner_csv" ] && [ -f "$outer_csv" ]; then
            ./a.out "$vals_csv" "$inner_csv" "$outer_csv" "$folder_name" "$a"
            ((a++))
        else
            echo "Error: CSV files not found in folder $folder_name."
        fi
    fi
done

# Clean up compiled executable
rm a.out

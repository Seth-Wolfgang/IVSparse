#!/bin/bash
# clear
# Step 1: Compile simulatedBench_CSC.cpp with g++ -O2
#  g++ -O2 -I ~/eigen simulatedBench_COO.cpp -o a.out                                                   # may need to remove -I ~/eigen

# Step 2: Iterate through folders in ~/matrices



matrix_dir="/home/sethwolfgang/matrices"                                                                     # Change this to your directory




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

        # Calculate the values
        ROWS=$(wc -l "$inner_csv" | awk '{print $1}')
        COLS=$(($(wc -l "$outer_csv" | awk '{print $1}') - 1))                                # starter code for changing the number of columns, rows, etc for the benchmarking code. I don't know if it works
        NNZ=$(wc -l "$vals_csv" | awk '{print $1}')

        # Search and replace the values in the benchmarking code
        input_file="simulatedBench_COO.cpp"
        sed -i "/#define ROWS/c\#define ROWS $ROWS" "$input_file"
        sed -i "/#define COLS/c\#define COLS $COLS" "$input_file"
        sed -i "/#define NNZ/c\#define NNZ $NNZ" "$input_file"

        g++ -O2 -I ~/eigen simulatedBench_COO.cpp -o a.out                                             # may need to remove -I ~/eigen                              


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

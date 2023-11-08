
## Simulated Data
Most plots listed in the publication related to this paper can be recreated by running the benchmarking
script. 
Timing results may vary due to differences in hardware, but compression sizes will remain the same.
To create plots, we used matplotlib to create plots generated from data produced by running ivsparse-benchmark.sh

Default paramters for simulated benchmarks:
ROWS = 1'000'000 -> handled by slurm script
COLS = 25

For ivsparse-benchmark.sh,

*   #define NUM_ITERATIONS 5
*   #define DENSITY 0.01 -> handled by slurm script
*   #define VALUE_TYPE float

The provided slurm script (ivsparse-benchmark.sh) should create the necessary files. 
Some work, mainly creating directories in /lib/, may need to be done for a generalized run, as the script is written for our specific machine.
This slurm script calls a python program, which in turn generates the random sparse matrices, temporarily saves them,
adjusts the underlying cpp file, compiles the cpp code, then runs the cpp code before moving onto the next matrix.
To run ivsparse-benchmark.sh, it is necessary to adjust the slurm script for use on whatever
cluster you have available.  

Running the slurm script will produce a large number of output csv files (one for each size,
for each format).  Once all data is obtained, the plotting scripts in
`Benchmarking/lib/plotting/` will reproduce the plots in the paper.

## Real Data -> see Table 1 in publication

To run the real data benchmarks, you will need to download the data from the following link: 

Web of Science: https://paperswithcode.com/dataset/web-of-science-dataset
    - Note: This data was run through a count vectorizer, the script is provided in helpful_scripts/
            
            
MovieLens: https://grouplens.org/datasets/movielens/
    - Note: We used the ratings.csv file, as part of their 25 million data set. 

Single-cell: https://www.10xgenomics.com/resources/datasets/aggregate-of-900k-human-non-small-cell-lung-cancer-and-normal-adjacent-cells-multiplexed-samples-16-probe-barcodes-1-standard
    - Note: We used filtered_feature_bc_matrix.mtx and 


PR02R: https://www.cise.ufl.edu/research/sparse/matrices/Fluorem/PR02R.html
    - Note: We used the PR02R.mtx NOT PR02R_b.mtx or PR02R_x.mtx.

com-orkut: https://sparse.tamu.edu/SNAP/com-Orkut

Once downloaded, the scripts in `Benchmarking/lib/real-problems-mem-testing` will produce a `data.csv`
file containing all of the necessary data to reproduce the table.
The slurm script `Benchmarking/lib/real-problems-mem-testing/run-mem-calc.sh` will run all of
the real datasets.

runBenchmark.sh is not used in the publication, but allows for benchmarkings using matrices from SuiteSparse Matrix Collection.
Early benchmarks used this, but we found the prevalence of diagonal matrices in the collection to be problematic for our data structure. 
IVSparse is not meant to be used for diagonal matrices, and so we do not include these results in our benchmarks. We did not create ssget, but 
a ours is a modifed version to store matrices in /Benchmarking/matrices. A link to the original repo/creator is provided at the top of the file.

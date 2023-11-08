/**
 * @file densityBenchmark.cpp
 * @author Seth Wolfgang
 * @brief This is meant to create a matrix of a set redundancy and track memory usage based on sparsity (# of nonzeros / rows * cols).
 *        This is used to make the second size subfigure in the publication associated with this library. To reproduce the results, compile 
 *        and run this file with the defualt arguments:
 *        #define NUM_ITERATIONS 10
 *        #define REDUNDANCY 0.1
 *        #define MATRICES 1000
 *        #define VALUE_TYPE double
 * 
 *        Performance benchmarks are included, but were never used in the publication, so only size ones will run by default. 
 *        You may uncomment performance benchmarks if you wish to run them.
 * 
 *       To create a plot of the results, compile and run this file and use the output data in the .Rmd file in the /R folder:
 * 
  *      Run the R script in the R folder. Look for simulated_bench_visualizations.Rmd in /Benchmarking/R.
 *       - This will create the plots used in the paper.
 *       - Some additional packages may be required for the Rmd file. See Rmd file for what you may need to install.
 *         Look for the library() calls.
 *       - You may need to change the path of where plots are saved, or you may comment out dev.off() and pdf() calls.
 *       - It is enough to simply run all cells in the rmd file if you ran the program with runFullSimulatedBench.sh
 * 
 * @version 1
 * @date 2023-08-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <chrono> 
#include "../../IVSparse/SparseMatrix"
#include <unordered_set>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iterator>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <random>



#define NUM_ITERATIONS 10
#define REDUNDANCY 0.1
#define MATRICES 1000
#define VALUE_TYPE double

template <typename T, typename indexType, int compressionLevel> double averageRedundancy(IVSparse::SparseMatrix<T, indexType, compressionLevel>& matrix);
template <typename T, typename indexType, int compressionLevel> double averageRedundancy(Eigen::SparseMatrix<T>& matrix);
template <int index> int __attribute__((optimize("Ofast"))) getMax(std::vector<std::tuple<int, int, VALUE_TYPE>>& data);
void printDataToFile(std::vector<uint64_t>& data, std::vector<std::vector<uint64_t>>& timeData, const char* filename);
void generateMatrix(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int numRows, int numCols, uint64_t seed);
inline __attribute__((optimize("O2"))) void adjustCoords(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int maxValue, int seed, int numRows, int numCols);
inline void adjustValues(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int maxValue, int seed);

void loadMatrix(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, char* filename);

void VCSC_Benchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int numRows, int numCols);
void IVCSC_Benchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int numRows, int numCols);
void eigen_Benchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int numRows, int numCols);

void VCSC_outerSumBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<uint64_t>>& resultData);
void VCSC_CSCConstructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<uint64_t>>& resultData, int rows, int cols);
void VCSC_constructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<uint64_t>>& resultData, int rows, int cols);
void VCSC_scalarBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2> matrix, std::vector<std::vector<uint64_t>>& resultData);
void VCSC_spmvBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numCols);
void VCSC_spmmBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols);

void IVCSC_outerSumBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<uint64_t>>& resultData);
void IVCSC_CSCConstructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<uint64_t>>& resultData, int rows, int cols);
void IVCSC_constructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<uint64_t>>& resultData, int rows, int cols);
void IVCSC_scalarBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3> matrix, std::vector<std::vector<uint64_t>>& resultData);
void IVCSC_spmvBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numCols);
void IVCSC_spmmBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols);

void eigen_outerSumBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<uint64_t>>& resultData);
void eigen_constructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<uint64_t>>& resultData, int rows, int cols);
void eigen_scalarBenchmark(Eigen::SparseMatrix<VALUE_TYPE> matrix, std::vector<std::vector<uint64_t>>& resultData);
void eigen_spmvBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numCols);
void eigen_spmmBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols);

int main(int argc, char** argv) {

    if (argc != 3) {
        argv = (char**)malloc(3 * sizeof(char*));
        argv[1] = (char*)malloc(10 * sizeof(char));
        argv[2] = (char*)malloc(10 * sizeof(char));
        argv[1] = "10000";
        argv[2] = "100";
    }
    std::cout << "Running with " << argv[1] << " rows and " << argv[2] << " columns" << std::endl;

    bool readFromDisk = false;
    std::vector<std::tuple<int, int, VALUE_TYPE>> coords;
    int rows, cols;
    if (readFromDisk) {
        char* filePath = "";
        loadMatrix(coords, filePath);
        rows = getMax<0>(coords);
        cols = getMax<1>(coords);
    }
    else {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        generateMatrix(coords, atoi(argv[1]), atoi(argv[2]), 1);
    }

    std::cout << "\033[34;42;1;4mStarting VCSC Benchmark\033[0m" << std::endl;
    VCSC_Benchmark(coords, atoi(argv[1]), atoi(argv[2]));
    std::cout << "\033[34;42;1;4mStarting IVCSC Benchmark\033[0m" << std::endl;
    IVCSC_Benchmark(coords, atoi(argv[1]), atoi(argv[2]));
    std::cout << "\033[34;42;1;4mStarting Eigen Benchmark\033[0m" << std::endl;
    eigen_Benchmark(coords, atoi(argv[1]), atoi(argv[2]));

    return 1;
}

/*************************************************************************************************************
 *                                                                                                           *
 *                                                                                                           *
 *                                           Helper Functions                                                *
 *                                                                                                           *
 *                                                                                                           *
 *                                                                                                           *
 ************************************************************************************************************/

void loadMatrix(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, char* filename) {
    std::unordered_map<std::string, double> map;

    FILE* file;
    file = fopen(filename, "r");
    char line[1024];

    uint i = 0;
    while (fgets(line, 1024, file)) {
        //split the row into 3 seperate values
        char* val1 = strtok(line, ",");
        char* val2 = strtok(NULL, ",");
        char* val3 = strtok(NULL, ",");
        std::string val3String(val3);

        //add the values to the data vector
        if constexpr (std::is_same<VALUE_TYPE, double>::value)
            data.push_back(std::make_tuple(atoi(val2), atoi(val1), atof(val3)));
        else
            data.push_back(std::make_tuple(atoi(val2), atoi(val1), atoi(val3)));

    }

    //copy data to data vector
    fclose(file);
}
/**
 * @brief gets the max in the tuple
 *
 *
 */

template <int index>
int __attribute__((optimize("Ofast"))) getMax(std::vector<std::tuple<int, int, VALUE_TYPE>>& data) {
    int max = 0;
    for (int i = 0; i < data.size(); i++) {
        if (std::get<index>(data.at(i)) > max) {
            max = std::get<index>(data.at(i));
        }
    }
    return max;
}

/**
 * @brief Prints results to CSV
 *
 * @param data
 * @param filename
 */

void printDataToFile(std::vector<double>& data, std::vector<std::vector<uint64_t>>& timeData, const char* filename) {
    FILE* file;

    //check if file exists
    int fileExists = access(filename, F_OK);
    if (fileExists == -1) {
        file = fopen(filename, "a");
        fprintf(file, "%s\n", "ID,rows,cols,nonzeros,sparsity,redundancy,size,constructor_time,scalar_time,spmv_time,spmm_time,sum_time");
        fclose(file);
    }
    file = fopen(filename, "a");

    for (uint64_t i = 0; i < timeData[i].size(); i++) {
        printf("Sparsity: %2.3lf | ", ((double)data.at(3) / (double)(data.at(1) * data.at(2))));
        // std::cout << "Sparsity: " << ( data.at(0) * data.at(1) /  data.at(3) ) << " | " ;
        std::cout << data.at(4) << " ";
        std::cout << timeData.at(i).at(0) << " " << timeData.at(i).at(1) << " " << timeData.at(i).at(2) << " " << timeData.at(i).at(3) << " " << timeData.at(i).at(4) << std::endl;
    }

    /**
     * ID, rows, cols, nonzeros, sparsity, redundancy, size,
     *
     * scalar time, spmv time, spmm time
     */

    for (uint64_t i = 0; i < timeData.size(); i++) {
        fprintf(file, "%.0lf, %.0lf, %.0lf, %lf, %lf, %lf, %lf, %lu, %lu, %lu, %lu, %lu\n", data.at(0), data.at(1), data.at(2), data.at(3), ((double)data.at(3) / (double)(data.at(1) * data.at(2))),
                data.at(4), data.at(5), timeData.at(i).at(0), timeData.at(i).at(1), timeData.at(i).at(2), timeData.at(i).at(3), timeData.at(i).at(4));

    }
    fclose(file);
}

void __attribute__((optimize("Ofast"))) generateMatrix(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int numRows, int numCols, uint64_t seed) {
    std::mt19937_64 rng(seed);
    uint numElements = static_cast<uint>(numRows * numCols);
    std::map<std::tuple<int, int, VALUE_TYPE>, bool> visited;  // Store visited coordinates

    while (data.size() <= numElements) {
        int row = rng() % numRows;
        int col = rng() % numCols;

        std::tuple<int, int, VALUE_TYPE> coordinate(row, col, fmod(rng(), numRows * REDUNDANCY) + 1);

        // Check if coordinate is already visited
        if (visited[coordinate]) {
            continue;  // Skip duplicate coordinates
        }

        visited[coordinate] = true;  // Mark coordinate as visited
        data.push_back(coordinate);
    }

    data.resize(numElements);
}

inline __attribute__((optimize("O2"))) void adjustValues(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int maxValue, int seed) {
    srand(seed);
    for (uint64_t i = 0; i < data.size(); i++) {
        std::get<2>(data.at(i)) = rand() % maxValue + 1;
    }
}



/**
 * @brief
 *
 * @tparam T
 * @param data
 * @param maxValue
 * @param seed
 */

inline __attribute__((optimize("O2"))) void adjustCoords(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int maxValue, int seed, int numRows, int numCols) {
    std::mt19937_64 rng(seed);
    std::map<std::tuple<int, int, VALUE_TYPE>, bool> visited;  // Store visited coordinates
    data.clear();

    while (data.size() < maxValue) {
        int row = rng() % numRows;
        int col = rng() % numCols;

        std::tuple<int, int, VALUE_TYPE> coordinate(row, col, 1);

        // Check if coordinate is already visited
        if (visited[coordinate]) {
            continue;  // Skip duplicate coordinates
        }

        visited[coordinate] = true;  // Mark coordinate as visited
        data.push_back(coordinate);
    }
    std::cout << "data size: " << data.size() << std::endl;

    data.resize(maxValue);
}

/**
 * @brief function to help control all VCSC benchmarks
 *
 * @param data
 * @param numRows
 * @param numCol
 * @param nonzeros
 */

void  VCSC_Benchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int numRows, int numCols) {
    std::vector<double> matrixData(1);
    std::vector<std::vector<uint64_t>> timeData(NUM_ITERATIONS);
    // adjustCoords(data, 1, 1, numRows, numCols);
    for (int i = 1; i <= MATRICES; i++) {
        int numValues = static_cast<int>(numRows * numCols * (double)((double)(i) / (double)MATRICES));
        adjustCoords(data, numValues, i, numRows, numCols);
        adjustValues(data, numValues * REDUNDANCY / numCols, i);
        IVSparse::SparseMatrix<VALUE_TYPE, int, 2> matrix(data, numRows, numCols, data.size());

        matrixData.resize(6);
        matrixData.at(0) = i;
        matrixData.at(1) = numRows;
        matrixData.at(2) = numCols;
        matrixData.at(3) = data.size();
        matrixData.at(4) = averageRedundancy<VALUE_TYPE, int, 2>(matrix);
        matrixData.at(5) = matrix.byteSize();

        for (int j = 0; j < NUM_ITERATIONS; j++) {
            timeData.at(j).resize(5);
        }

        // VCSC_constructorBenchmark(data, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": VCSC constructor done" << std::endl;
        // VCSC_scalarBenchmark(matrix, timeData);
        // std::cout << i << "/" << MATRICES << ": VCSC scalar done" << std::endl;
        // VCSC_outerSumBenchmark(matrix, timeData);
        // std::cout << i << "/" << MATRICES << ": VCSC column sums done" << std::endl;
        // VCSC_spmvBenchmark(matrix, timeData, numCols);
        // std::cout << i << "/" << MATRICES << ": VCSC spmv done" << std::endl;
        // VCSC_spmmBenchmark(matrix, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": VCSC spmm done\n" << std::endl;

        std::stringstream path;
        path << "../results/density_VCSCResults_" << REDUNDANCY << ".csv";
        printDataToFile(matrixData, timeData, path.str().c_str());
        data.clear();

    }
}

/**
 * @brief Function to help control all IVCSC benchmarks
 *
 * @param data
 * @param numRows
 * @param numCol
 * @param nonzeros
 */

void   IVCSC_Benchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int numRows, int numCols) {
    std::vector<double> matrixData(1);
    std::vector<std::vector<uint64_t>> timeData(NUM_ITERATIONS);
    // adjustCoords(data, 1, 1, numRows, numCols);

    for (int i = 1; i <= MATRICES; i++) {
        int numValues = static_cast<int>(numRows * numCols * (double)((double)(i) / (double)MATRICES));
        adjustCoords(data, numValues, i, numRows, numCols);
        adjustValues(data, numValues / numRows * REDUNDANCY + 1, i);
        IVSparse::SparseMatrix<VALUE_TYPE, int, 3> matrix(data, numRows, numCols, data.size());

        matrixData.resize(6);
        matrixData.at(0) = i;
        matrixData.at(1) = numRows;
        matrixData.at(2) = numCols;
        matrixData.at(3) = data.size();
        matrixData.at(4) = averageRedundancy<VALUE_TYPE, int, 3>(matrix);
        matrixData.at(5) = matrix.byteSize();

        for (int j = 0; j < NUM_ITERATIONS; j++) {
            timeData.at(j).resize(5);
        }

        // IVCSC_constructorBenchmark(data, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": IVCSC constructor done" << std::endl;
        // IVCSC_scalarBenchmark(matrix, timeData);
        // std::cout << i << "/" << MATRICES << ": IVCSC scalar done" << std::endl;
        // IVCSC_outerSumBenchmark(matrix, timeData);
        // std::cout << i << "/" << MATRICES << ": IVCSC column sums done" << std::endl;
        // IVCSC_spmvBenchmark(matrix, timeData, numCols);
        // std::cout << i << "/" << MATRICES << ": IVCSC spmv done" << std::endl;
        // IVCSC_spmmBenchmark(matrix, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": IVCSC spmm done\n" << std::endl;

        std::stringstream path;
        path << "../results/density_IVCSCResults_" << REDUNDANCY << ".csv";
        printDataToFile(matrixData, timeData, path.str().c_str());
        data.clear();
        // adjustCoords(data, static_cast<int>(numRows * numCols * ((double)(MATRICES - i) / (double)MATRICES)), i, numRows, numCols);
    }
}

/**
 * @brief function to help cntrol all eigen benchmarks
 *
 * @param data
 * @param numRows
 * @param numCol
 * @param nonzeros
 */

void eigen_Benchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int numRows, int numCols) {
    std::vector<Eigen::Triplet<VALUE_TYPE>> triplet;
    std::vector<double> matrixData(1);
    std::vector<std::vector<uint64_t>> timeData(NUM_ITERATIONS);

    for (int i = 1; i <= MATRICES; i++) {
        int numValues = static_cast<int>(numRows * numCols * (double)((double)(i) / (double)MATRICES));
        std::cout << "numValues: " << numValues << std::endl;
        adjustCoords(data, numValues, i, numRows, numCols);
        adjustValues(data, numValues / numRows * REDUNDANCY + 1, i);
        for (int i = 0; i < data.size(); i++) {
            triplet.push_back(Eigen::Triplet<VALUE_TYPE>(std::get<0>(data.at(i)), std::get<1>(data.at(i)), std::get<2>(data.at(i))));
        }

        Eigen::SparseMatrix<VALUE_TYPE> matrix(numRows, numCols);
        matrix.reserve(data.size());
        matrix.setFromTriplets(triplet.begin(), triplet.end());


        matrixData.resize(6);
        matrixData.at(0) = i;
        matrixData.at(1) = numRows;
        matrixData.at(2) = numCols;
        matrixData.at(3) = data.size();
        matrixData.at(4) = averageRedundancy<VALUE_TYPE, int, 2>(matrix);


        matrixData.at(5) = (double)(matrix.nonZeros() * sizeof(VALUE_TYPE) + matrix.nonZeros() * sizeof(uint32_t) + (101) * sizeof(uint32_t));
        std::cout << "matrix nonzeros: " << matrix.nonZeros() << std::endl;

        printf("Ratio: %lf | Eigen size: %lf | Dense size %d\n", (matrixData.at(5) / (10000 * 100 * 8)), matrixData.at(5), 10000 * 100 * 8);

        for (int j = 0; j < NUM_ITERATIONS; j++) {
            timeData.at(j).resize(5);
        }

        // eigen_constructorBenchmark(data, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": Eigen constructor done" << std::endl;
        // eigen_scalarBenchmark(matrix, timeData);
        // std::cout << i << "/" << MATRICES << ": Eigen scalar done" << std::endl;
        // eigen_outerSumBenchmark(matrix, timeData);
        // std::cout << i << "/" << MATRICES << ": Eigen column sums done" << std::endl;
        // eigen_spmvBenchmark(matrix, timeData, numCols);
        // std::cout << i << "/" << MATRICES << ": Eigen spmv done" << std::endl;
        // eigen_spmmBenchmark(matrix, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": Eigen spmm done" << std::endl;

        std::stringstream path;
        path << "../results/density_EigenResults_" << REDUNDANCY << ".csv";
        printDataToFile(matrixData, timeData, path.str().c_str());
        triplet.clear();

        // adjustCoords(data, static_cast<int>(numRows * numCols * ((double)(MATRICES - i) / (double)MATRICES)), i, numRows, numCols);
    }
}

/*************************************************************************************************************
 *                                                                                                           *
 *                                                                                                           *
 *                                           Performance Benchmarks                                          *
 *                                                                                                           *
 *                                                                                                           *
 *                                                                                                           *
 ************************************************************************************************************/

 /**
  * @brief Benchmark for VCSC constructor
  *
  * @param data
  * @param resultData
  * @param rows
  * @param cols
  */

void   VCSC_constructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<uint64_t>>& resultData, int rows, int cols) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    //cold start
    for (int i = 0; i < 1; i++) {
        IVSparse::SparseMatrix<VALUE_TYPE, int, 2> matrix(data, rows, cols, data.size());
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        IVSparse::SparseMatrix<VALUE_TYPE, int, 2> matrix(data, rows, cols, data.size());
        end = std::chrono::high_resolution_clock::now();
        resultData.at(i).at(0) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

}

/**
 * @brief Benchmark for VCSC scalar multiplication
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void   VCSC_scalarBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2> matrix, std::vector<std::vector<uint64_t>>& resultData) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    IVSparse::SparseMatrix<VALUE_TYPE, int, 2> result;

    //cold start
    for (int i = 0; i < 1; i++) {
        result = matrix * 2;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        matrix *= 2;
        end = std::chrono::high_resolution_clock::now();
        resultData.at(i).at(1) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
}

/**
 * @brief Benchmark for VCSC Matrix * (dense) vector multiplication
 *
 * @param matrix
 * @param numCols
 */

void   VCSC_spmvBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numCols) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    Eigen::Matrix<VALUE_TYPE, -1, 1> eigenVector = Eigen::Matrix<VALUE_TYPE, -1, 1>::Random(numCols);
    Eigen::Matrix<VALUE_TYPE, -1, 1> result;

    //cold start
    for (int i = 0; i < 1; i++) {
        result = matrix * eigenVector;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = matrix * eigenVector;
        end = std::chrono::high_resolution_clock::now();
        resultData.at(i).at(2) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

}

/**
 * @brief Benchmark for VCSC Matrix * Matrix multiplication
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void VCSC_spmmBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols) {
    Eigen::Matrix<VALUE_TYPE, -1, -1> eigenMatrix = Eigen::Matrix<VALUE_TYPE, -1, -1>::Random(numCols, numRows);
    Eigen::Matrix<VALUE_TYPE, -1, -1> result;
    std::chrono::time_point<std::chrono::system_clock> start, end;

    //cold start
    for (int i = 0; i < 1; i++) {
        result = matrix * eigenMatrix;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = matrix * eigenMatrix;
        end = std::chrono::high_resolution_clock::now();
        resultData.at(i).at(3) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
}

/**
 * @brief Benchmark for column sums
 *
 * @param matrix
 * @param resultData
 */

void VCSC_outerSumBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<uint64_t>>& resultData) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    IVSparse::SparseMatrix<VALUE_TYPE, int, 3> result;
    int sum;

    //cold start
    for (int i = 0; i < 1; i++) {
        sum = matrix.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        sum = matrix.sum();
        end = std::chrono::high_resolution_clock::now();
        resultData.at(i).at(4) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    // std::cout << "Sum: " << sum << std::endl;
}



/**
 * @brief Benchmark for IVCSC constructor
 *
 * @param data
 * @param resultData
 * @param rows
 * @param cols
 */

void IVCSC_constructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<uint64_t>>& resultData, int rows, int cols) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    //cold start
    for (int i = 0; i < 1; i++) {
        IVSparse::SparseMatrix<VALUE_TYPE, int, 3> matrix(data, rows, cols, data.size());
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        IVSparse::SparseMatrix<VALUE_TYPE, int, 3> matrix(data, rows, cols, data.size());
        end = std::chrono::high_resolution_clock::now();
        resultData.at(i).at(0) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
}

/**
 * @brief Benchmark for IVCSC scalar multiplication
 *
 * @param matrix
 */

void IVCSC_scalarBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3> matrix, std::vector<std::vector<uint64_t>>& resultData) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    IVSparse::SparseMatrix<VALUE_TYPE, int, 3> result;

    //cold start
    for (int i = 0; i < 1; i++) {
        result = matrix * 2;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        matrix *= 2;
        end = std::chrono::high_resolution_clock::now();
        resultData.at(i).at(1) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
}

/**
 * @brief Benchmark for Sparse Matrix * (dense) Vector
 *
 * @param matrix
 * @param resultData
 * @param numCols
 */

void   IVCSC_spmvBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numCols) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    Eigen::Matrix<VALUE_TYPE, -1, 1> eigenVector = Eigen::Matrix<VALUE_TYPE, -1, 1>::Random(numCols);
    Eigen::Matrix<VALUE_TYPE, -1, 1> result;

    //cold start
    for (int i = 0; i < 1; i++) {
        result = matrix * eigenVector;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = matrix * eigenVector;
        end = std::chrono::high_resolution_clock::now();
        resultData.at(i).at(2) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
}


/**
 * @brief Benchmark for IVCSC Matrix * (dense) Matrix multiplication
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void IVCSC_spmmBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols) {
    Eigen::Matrix<VALUE_TYPE, -1, -1> eigenMatrix = Eigen::Matrix<VALUE_TYPE, -1, -1>::Random(numCols, numRows);
    Eigen::Matrix<VALUE_TYPE, -1, -1> result;
    std::chrono::time_point<std::chrono::system_clock> start, end;

    //cold start
    for (int i = 0; i < 1; i++) {
        result = matrix * eigenMatrix;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = matrix * eigenMatrix;
        end = std::chrono::high_resolution_clock::now();
        resultData.at(i).at(3) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
}

/**
 * @brief  Benchmark for IVCSC Matrix column sums
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */


void IVCSC_outerSumBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<uint64_t>>& resultData) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    IVSparse::SparseMatrix<VALUE_TYPE, int, 3> result;
    int sum;

    //cold start
    for (int i = 0; i < 1; i++) {
        sum = matrix.sum();
    }


    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        sum = matrix.sum();
        end = std::chrono::high_resolution_clock::now();
        resultData.at(i).at(4) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    std::cout << "Sum: " << sum << std::endl;

}


/**
 * @brief Benchmark for Eigen constructor
 *
 * @param data
 * @param resultData
 * @param rows
 * @param cols
 */

void   eigen_constructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<uint64_t>>& resultData, int rows, int cols) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    Eigen::SparseMatrix<VALUE_TYPE> matrix(rows, cols);
    std::vector<Eigen::Triplet<VALUE_TYPE>> triplet;

    for (int i = 0; i < data.size(); i++) {
        triplet.push_back(Eigen::Triplet<VALUE_TYPE>(std::get<0>(data.at(i)), std::get<1>(data.at(i)), std::get<2>(data.at(i))));
    }

    //cold start
    for (int i = 0; i < 1; i++) {
        Eigen::SparseMatrix<VALUE_TYPE> matrix(rows, cols);
        matrix.setFromTriplets(triplet.begin(), triplet.end());
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();

        Eigen::SparseMatrix<VALUE_TYPE> matrix(rows, cols);
        matrix.setFromTriplets(triplet.begin(), triplet.end());

        end = std::chrono::high_resolution_clock::now();
        resultData.at(i).at(0) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
}

/**
 * @brief Benchmark for Eigen scalar multiplication
 *
 * @param matrix
 */

void   eigen_scalarBenchmark(Eigen::SparseMatrix<VALUE_TYPE> matrix, std::vector<std::vector<uint64_t>>& resultData) {
    Eigen::SparseMatrix<VALUE_TYPE> result;
    std::chrono::time_point<std::chrono::system_clock> start, end;

    //cold start
    for (int i = 0; i < 1; i++) {
        matrix *= 2;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        matrix *= 2;
        end = std::chrono::high_resolution_clock::now();
        // std::cout << result.sum() << std::endl;
        resultData.at(i).at(1) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
}

/**
 * @brief Benchmark for Eigen Matrix * vector multiplication
 *
 * @param matrix
 * @param numRows
 */

void eigen_spmvBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numCols) {
    Eigen::Matrix<VALUE_TYPE, -1, 1> eigenVector = Eigen::Matrix<VALUE_TYPE, -1, 1>::Random(numCols);
    Eigen::Matrix<VALUE_TYPE, -1, 1> result;
    std::chrono::time_point<std::chrono::system_clock> start, end;

    //cold start
    for (int i = 0; i < 1; i++) {
        result = matrix * eigenVector;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = matrix * eigenVector;
        end = std::chrono::high_resolution_clock::now();
        // std::cout << result << std::endl;
        resultData.at(i).at(2) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    // for(int i = 0; i < resultData.size(); i++){
    //     std::cout << resultData.at(i).at(2) << std::endl;
    // }
}

/**
 * @brief Benchmark for Eigen Matrix * Matrix multiplication
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void eigen_spmmBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols) {
    Eigen::Matrix<VALUE_TYPE, -1, -1> eigenMatrix = Eigen::Matrix<VALUE_TYPE, -1, -1>::Random(numCols, numRows);
    Eigen::Matrix<VALUE_TYPE, -1, -1> result;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    //cold start
    for (int i = 0; i < 1; i++) {
        result = matrix * eigenMatrix;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = matrix * eigenMatrix;
        end = std::chrono::high_resolution_clock::now();
        resultData.at(i).at(3) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
}

/**
 * @brief Benchmark for column sums
 *
 * @param matrix
 * @param resultData
 */

void eigen_outerSumBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<uint64_t>>& resultData) {
    Eigen::SparseMatrix<VALUE_TYPE> result;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    int sum;

    //cold start
    for (int i = 0; i < 1; i++) {
        sum = matrix.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        sum = matrix.sum();
        end = std::chrono::high_resolution_clock::now();
        // std::cout << result.sum() << std::endl;
        resultData.at(i).at(4) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    std::cout << "Sum: " << sum << std::endl;

}


template <typename T, typename indexType, int compressionLevel>
double averageRedundancy(IVSparse::SparseMatrix<T, indexType, compressionLevel>& matrix) {
    int numRows = matrix.rows();
    int numCols = matrix.cols();
    int colsWithValues = 0;
    double totalRedundancy = 0.0;

    for (int j = 0; j < numCols; ++j) {
        double totalValues = 0;
        double redundancy = 0;
        std::unordered_map<double, double> uniqueValues;
        for (typename IVSparse::SparseMatrix<T, indexType, compressionLevel>::InnerIterator it(matrix, j); it; ++it) {
            uniqueValues.insert(std::pair<double, int>(it.value(), 0));
            totalValues++;
        }
        if (totalValues == 0 || uniqueValues.size() == 0)
            continue;
        else if (uniqueValues.size() == 1)
            return 1;
        else {
            redundancy = 1 - (uniqueValues.size() / totalValues);
        }
        totalRedundancy += redundancy;
        colsWithValues++;
    }

    return totalRedundancy / static_cast<double>(colsWithValues);
}

template <typename T, typename indexType, int compressionLevel>
double averageRedundancy(Eigen::SparseMatrix<T>& matrix) {
    const int numRows = matrix.rows();
    const int numCols = matrix.cols();
    int colsWithValues = 0;
    double totalRedundancy = 0.0;

    for (int j = 0; j < numCols; ++j) {
        double totalValues = 0;
        double redundancy = 0;
        std::unordered_map<double, double> uniqueValues;

        for (typename Eigen::SparseMatrix<T>::InnerIterator it(matrix, j); it; ++it) {
            uniqueValues.insert(std::pair<double, int>(it.value(), 0));
            totalValues++;
        }

        if (totalValues == 0 || uniqueValues.size() == 0)
            continue;
        else if (uniqueValues.size() == 1)
            return 1;
        else
            redundancy = 1 - (uniqueValues.size() / totalValues);
        totalRedundancy += redundancy;
        colsWithValues++;
    }
    return totalRedundancy / static_cast<double>(colsWithValues);
}

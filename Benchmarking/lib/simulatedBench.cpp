/**
 * @file simulatedBench.cpp
 * @author Seth Wolfgang
 * @brief This benchmark is the one used ot create data for plots in the paper.
 *        It is meant to be run with a specified number of rows, columns, and a specified density.
 *        Redundancy is adjusted each iteration, and the change is modified by the defined MATRICES variable.
 *        Less matrices leads to bigger changes in redundancy between iterations. Data from this can be plotted
 *        using the supplied R script in the R folder. Look for simulated_bench_visualizations.Rmd in /Benchmarking/R.
 * @version 1
 * @date 2023-08-30
 *
 * @copyright Copyright (c) 2023
 *
 *
 * STEPS FOR REPRODUCABILITY:
 * 1. Run this program using runFullSimulatedBench.sh with the desired number of rows and columns, and the desired density.
 *  - The paper uses 10,000x100 matrices
 *  - The following should be defined as such:
    NUM_ITERATIONS 10
    DENSITY 0.01 -> THIS WILL CHANGE IF YOU RUN WITH runFullSimulatedBench.sh. This is intentional.
    MATRICES 1000
    VALUE_TYPE double

 *
 * 2. Run the R script in the R folder. Look for simulated_bench_visualizations.Rmd in /Benchmarking/R.
 *  - This will create the plots used in the paper.
 *  - Some additional packages may be required for the Rmd file. See Rmd file for what you may need to install.
 *    Look for the library() calls.
 *  - You may need to change the path of where plots are saved, or you may comment out dev.off() and pdf() calls.
 *  - It is enough to simply run all cells in the rmd file if you ran the program with runFullSimulatedBench.sh
 *
 *
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
#define NUM_COLD_STARTS 1
#define DENSITY 0.05
#define MATRICES 1000
#define VALUE_TYPE double

template <typename T, typename indexType, int compressionLevel> double averageRedundancy(IVSparse::SparseMatrix<T, indexType, compressionLevel>& matrix);
template <typename T, typename indexType, int compressionLevel> double averageRedundancy(Eigen::SparseMatrix<T>& matrix);
template <int index> int __attribute__((optimize("Ofast"))) getMax(std::vector<std::tuple<int, int, VALUE_TYPE>>& data);
void printDataToFile(std::vector<uint64_t>& data, std::vector<std::vector<uint64_t>>& timeData, const char* filename);
void generateMatrix(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int numRows, int numCols, uint64_t seed);
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
void VCSC_transposeBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols);
void VCSC_iteratorBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols);

void IVCSC_outerSumBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<uint64_t>>& resultData);
void IVCSC_CSCConstructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<uint64_t>>& resultData, int rows, int cols);
void IVCSC_constructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<uint64_t>>& resultData, int rows, int cols);
void IVCSC_scalarBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3> matrix, std::vector<std::vector<uint64_t>>& resultData);
void IVCSC_spmvBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numCols);
void IVCSC_spmmBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols);
void IVCSC_transposeBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols);
void IVCSC_iteratorBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols);

void eigen_outerSumBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<uint64_t>>& resultData);
void eigen_constructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<uint64_t>>& resultData, int rows, int cols);
void eigen_scalarBenchmark(Eigen::SparseMatrix<VALUE_TYPE> matrix, std::vector<std::vector<uint64_t>>& resultData);
void eigen_spmvBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numCols);
void eigen_spmmBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols);
void eigen_transposeBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols);
void eigen_iteratorBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols);

template <typename T> inline Eigen::Matrix<T, -1, -1> eigen_fair_spmm(Eigen::SparseMatrix<T>& leftMat, Eigen::Matrix<T, -1, -1>& rightMat);

template <typename T> inline Eigen::Matrix<T, -1, 1> eigen_fair_spmv(Eigen::SparseMatrix<T>& matrix, Eigen::Matrix<T, -1, 1>& vector);

template <typename T, typename indexType, int compressionLevel> inline Eigen::Matrix<T, -1, -1> IVSparse_fair_spmm(IVSparse::SparseMatrix<T, indexType, compressionLevel>& leftMat, Eigen::Matrix<T, -1, -1>& rightMat);

template <typename T, typename indexType, int compressionLevel> inline Eigen::Matrix<T, -1, 1> IVSparse_fair_spmv(IVSparse::SparseMatrix<T, indexType, compressionLevel>& matrix, Eigen::Matrix<T, -1, 1>& vector);

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

    // std::cout << "\033[34;42;1;4mStarting VCSC Benchmark\033[0m" << std::endl;
    // VCSC_Benchmark(coords, atoi(argv[1]), atoi(argv[2]));
    // std::cout << "\033[34;42;1;4mStarting IVCSC Benchmark\033[0m" << std::endl;
    IVCSC_Benchmark(coords, atoi(argv[1]), atoi(argv[2]));
    // std::cout << "\033[34;42;1;4mStarting Eigen Benchmark\033[0m" << std::endl;
    // eigen_Benchmark(coords, atoi(argv[1]), atoi(argv[2]));

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
        fprintf(file, "%s\n", "ID,rows,cols,nonzeros,sparsity,redundancy,size,constructor_time,scalar_time,spmv_time,spmm_time,sum_time,transpose_time,iterator_time");
        fclose(file);
    }
    file = fopen(filename, "a");

    for (uint64_t i = 0; i < timeData[i].size(); i++) {
        std::cout << data.at(4) << " ";
        std::cout << timeData.at(i).at(0) << " " << timeData.at(i).at(1) << " " << timeData.at(i).at(2) << " " <<
            timeData.at(i).at(3) << " " << timeData.at(i).at(4) << " " << timeData.at(i).at(5) << " " << timeData.at(i).at(6) << std::endl;
    }

    /**
     * ID, rows, cols, nonzeros, sparsity, redundancy, size,
     *
     * scalar time, spmv time, spmm time
     */
     // double redundancy = (double)(1.0 / data.at(3));

    for (uint64_t i = 0; i < timeData.size(); i++) {
        fprintf(file, "%.0lf, %.0lf, %.0lf, %lf, %lf, %lf, %lf, %lu, %lu, %lu, %lu, %lu, %lu, %lu\n", data.at(0), data.at(1), data.at(2), data.at(3), DENSITY,
                data.at(4), data.at(5), timeData.at(i).at(0), timeData.at(i).at(1), timeData.at(i).at(2), timeData.at(i).at(3), timeData.at(i).at(4),
                timeData.at(i).at(5), timeData.at(i).at(6));

    }
    fclose(file);
}

void __attribute__((optimize("O2"))) generateMatrix(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int numRows, int numCols, uint64_t seed) {
    std::mt19937_64 rng(seed);
    uint numElements = static_cast<uint>(numRows * numCols * DENSITY);
    std::map<std::tuple<int, int, VALUE_TYPE>, bool> visited;  // Store visited coordinates

    while (data.size() < numElements) {
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

    data.resize(numElements);
}



/**
 * @brief This randomizes values to help control redundancy between trials
 *
 * @tparam T
 * @param data
 * @param maxValue
 * @param seed
 */

inline __attribute__((optimize("O2"))) void adjustValues(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int maxValue, int seed) {
    srand(seed);
    for (uint64_t i = 0; i < data.size(); i++) {
        int val = rand() % maxValue + 1;
        std::get<2>(data.at(i)) = val;
    }


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
    IVSparse::SparseMatrix<VALUE_TYPE, int, 2> matrix;
    adjustValues(data, 1, 1);
    for (int i = 0; i < MATRICES; i++) {
        matrix = IVSparse::SparseMatrix<VALUE_TYPE, int, 2>(data, numRows, numCols, data.size());

        matrixData.resize(6);
        matrixData.at(0) = i;
        matrixData.at(1) = numRows;
        matrixData.at(2) = numCols;
        matrixData.at(3) = data.size();
        matrixData.at(4) = averageRedundancy<VALUE_TYPE, int, 2>(matrix);
        matrixData.at(5) = matrix.byteSize();

        for (int j = 0; j < NUM_ITERATIONS; j++) {
            timeData.at(j).resize(7);
        }

        // matrix.print();

        // VCSC_constructorBenchmark(data, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": VCSC constructor done" << std::endl;
        // VCSC_scalarBenchmark(matrix, timeData);
        // std::cout << i << "/" << MATRICES << ": VCSC scalar done" << std::endl;
        // VCSC_outerSumBenchmark(matrix, timeData);
        // std::cout << i << "/" << MATRICES << ": VCSC column sums done" << std::endl;
        VCSC_spmvBenchmark(matrix, timeData, numCols);
        std::cout << i << "/" << MATRICES << ": VCSC spmv done" << std::endl;
        // VCSC_spmmBenchmark(matrix, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": VCSC spmm done\n" << std::endl;
        // VCSC_iteratorBenchmark(matrix, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": VCSC iterator done\n" << std::endl;
        // VCSC_transposeBenchmark(matrix, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": VCSC transpose done\n" << std::endl;

        std::stringstream path;
        path << "../results/VCSCResults_" << DENSITY << ".csv";
        printDataToFile(matrixData, timeData, path.str().c_str());
        adjustValues(data, static_cast<int>(((double)numRows / (double)MATRICES) * (i + 1)), i);
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
    adjustValues(data, 1, 1);

    for (int i = 0; i < MATRICES; i++) {
        IVSparse::SparseMatrix<VALUE_TYPE, int, 3> matrix(data, numRows, numCols, data.size());

        matrixData.resize(6);
        matrixData.at(0) = i;
        matrixData.at(1) = numRows;
        matrixData.at(2) = numCols;
        matrixData.at(3) = data.size();
        matrixData.at(4) = averageRedundancy<VALUE_TYPE, int, 3>(matrix);
        matrixData.at(5) = matrix.byteSize();

        for (int j = 0; j < NUM_ITERATIONS; j++) {
            timeData.at(j).resize(7);
        }

        // IVCSC_constructorBenchmark(data, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": IVCSC constructor done" << std::endl;
        // IVCSC_scalarBenchmark(matrix, timeData);
        // std::cout << i << "/" << MATRICES << ": IVCSC scalar done" << std::endl;
        // IVCSC_outerSumBenchmark(matrix, timeData);
        // std::cout << i << "/" << MATRICES << ": IVCSC column sums done" << std::endl;
        IVCSC_spmvBenchmark(matrix, timeData, numCols);
        std::cout << i << "/" << MATRICES << ": IVCSC spmv done" << std::endl;
        // IVCSC_spmmBenchmark(matrix, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": IVCSC spmm done\n" << std::endl;
        // IVCSC_iteratorBenchmark(matrix, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": IVCSC iterator done\n" << std::endl;
        // IVCSC_transposeBenchmark(matrix, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": IVCSC transpose done\n" << std::endl;

        std::stringstream path;
        path << "../results/IVCSCResults_" << DENSITY << ".csv";
        printDataToFile(matrixData, timeData, path.str().c_str());
        adjustValues(data, static_cast<int>(((double)numRows / (double)MATRICES) * (i + 1)), i);

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
    Eigen::SparseMatrix<VALUE_TYPE> matrix(numRows, numCols);
    matrix.reserve(data.size());
    std::vector<Eigen::Triplet<VALUE_TYPE>> triplet;
    std::vector<double> matrixData(1);
    std::vector<std::vector<uint64_t>> timeData(NUM_ITERATIONS);
    adjustValues(data, 1, 1);

    for (int i = 0; i < MATRICES; i++) {
        triplet.clear();
        triplet.reserve(data.size());
        for (int j = 0; j < data.size(); j++) {
            triplet.push_back(Eigen::Triplet<VALUE_TYPE>(std::get<0>(data.at(j)), std::get<1>(data.at(j)), std::get<2>(data.at(j))));
        }
        matrix.setFromTriplets(triplet.begin(), triplet.end());

        matrixData.resize(6);
        matrixData.at(0) = i;
        matrixData.at(1) = numRows;
        matrixData.at(2) = numCols;
        matrixData.at(3) = data.size();
        matrixData.at(4) = averageRedundancy<VALUE_TYPE, int, 2>(matrix);
        matrixData.at(5) = matrix.nonZeros() * sizeof(VALUE_TYPE) + matrix.nonZeros() * sizeof(uint32_t) + (matrix.outerSize() + 1) * sizeof(uint32_t);

        for (int k = 0; k < NUM_ITERATIONS; k++) {
            timeData.at(k).resize(7);
        }

        // eigen_constructorBenchmark(data, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": Eigen constructor done" << std::endl;
        // eigen_scalarBenchmark(matrix, timeData);
        // std::cout << i << "/" << MATRICES << ": Eigen scalar done" << std::endl;
        // eigen_outerSumBenchmark(matrix, timeData);
        // std::cout << i << "/" << MATRICES << ": Eigen column sums done" << std::endl;
        eigen_spmvBenchmark(matrix, timeData, numCols);
        std::cout << i << "/" << MATRICES << ": Eigen spmv done" << std::endl;
        // eigen_spmmBenchmark(matrix, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": Eigen spmm done" << std::endl;
        // eigen_iteratorBenchmark(matrix, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": Eigen iterator done\n" << std::endl;
        // eigen_transposeBenchmark(matrix, timeData, numRows, numCols);
        // std::cout << i << "/" << MATRICES << ": Eigen transpose done\n" << std::endl;

        std::stringstream path;
        path << "../results/EigenResults_" << DENSITY << ".csv";
        printDataToFile(matrixData, timeData, path.str().c_str());
        adjustValues(data, static_cast<int>(((double)numRows / (double)MATRICES) * (i + 1)), i);

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
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
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
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = matrix * 2;
        std::cout << "sum: " << result.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        matrix *= 2;
        end = std::chrono::high_resolution_clock::now();
        std::cout << "sum: " << result.sum();
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
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = IVSparse_fair_spmv<VALUE_TYPE, int, 2>(matrix, eigenVector);
        std::cout << "sum: " << result.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = IVSparse_fair_spmv<VALUE_TYPE, int, 2>(matrix, eigenVector);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "sum: " << result.sum();
        resultData.at(i).at(2) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    std::cout << "sum: " << result.sum() << std::endl;
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
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = IVSparse_fair_spmm<VALUE_TYPE, int, 2>(matrix, eigenMatrix);
        std::cout << "sum: " << result.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = IVSparse_fair_spmm<VALUE_TYPE, int, 2>(matrix, eigenMatrix);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "sum: " << result.sum();
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
    IVSparse::SparseMatrix<VALUE_TYPE, int, 2> result;
    int sum;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        sum = matrix.sum();
        std::cout << "sum: " << sum;

    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        sum = matrix.sum();
        end = std::chrono::high_resolution_clock::now();
        std::cout << "sum: " << sum;

        resultData.at(i).at(4) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    // std::cout << "Sum: " << sum << std::endl;
}

/**
 * @brief Benchmark for VCSC Matrix Transpose
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void VCSC_transposeBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    IVSparse::SparseMatrix<VALUE_TYPE, int, 2> result;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = matrix.transpose();
        assert(result.sum() == matrix.sum());
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = matrix.transpose();
        end = std::chrono::high_resolution_clock::now();
        assert(result.sum() == matrix.sum());
        resultData.at(i).at(5) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
}

/**
 * @brief Benchmark for VCSC Matrix iterator traversal
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void VCSC_iteratorBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    int sum = 0;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        for (int j = 0; j < matrix.outerSize(); j++)
            for (IVSparse::SparseMatrix<VALUE_TYPE, int, 2>::InnerIterator it(matrix, j); it; ++it) {
                sum += *it;
            }
        std::cout << "sum: " << sum << std::endl;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < matrix.outerSize(); j++)
            for (IVSparse::SparseMatrix<VALUE_TYPE, int, 2>::InnerIterator it(matrix, j); it; ++it) {
                sum += *it;
            }

        end = std::chrono::high_resolution_clock::now();
        std::cout << "sum: " << sum << std::endl;
        resultData.at(i).at(6) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
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
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        IVSparse::SparseMatrix<VALUE_TYPE, int, 3> matrix(data, rows, cols, data.size());
        std::cout << "sum: " << matrix.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        IVSparse::SparseMatrix<VALUE_TYPE, int, 3> matrix(data, rows, cols, data.size());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "sum: " << matrix.sum();
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
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = matrix * 2;
        std::cout << "sum: " << result.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        matrix *= 2;
        end = std::chrono::high_resolution_clock::now();
        std::cout << "sum: " << result.sum();

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
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = IVSparse_fair_spmv<VALUE_TYPE, int, 3>(matrix, eigenVector);
        std::cout << "sum: " << result.sum();

    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = IVSparse_fair_spmv<VALUE_TYPE, int, 3>(matrix, eigenVector);
        end = std::chrono::high_resolution_clock::now();

        std::cout << "sum: " << result.sum();

        resultData.at(i).at(2) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    std::cout << "sum: " << (result.sum()) << std::endl;

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
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = IVSparse_fair_spmm<VALUE_TYPE, int, 3>(matrix, eigenMatrix);
        std::cout << "sum: " << result.sum() << std::endl;


    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = IVSparse_fair_spmm<VALUE_TYPE, int, 3>(matrix, eigenMatrix);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "sum: " << result.sum();

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
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        sum = matrix.sum();
        std::cout << "sum: " << sum;
    }


    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        sum = matrix.sum();
        end = std::chrono::high_resolution_clock::now();
        std::cout << "sum: " << sum;

        resultData.at(i).at(4) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

}



/**
 * @brief Benchmark for IVCSC Matrix Transpose
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void IVCSC_transposeBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    IVSparse::SparseMatrix<VALUE_TYPE, int, 3> result;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = matrix.transpose();
        assert(result.sum() == matrix.sum());
        std::cout << "sum: " << result.sum();

    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = matrix.transpose();
        end = std::chrono::high_resolution_clock::now();
        assert(result.sum() == matrix.sum());
        std::cout << "sum: " << result.sum();

        resultData.at(i).at(5) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
}

/**
 * @brief Benchmark for IVCSC Matrix iterator traversal
 *
 * @param matrix
 * @param resultData
 * @param numRows
 * @param numCols
 */

void IVCSC_iteratorBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    int sum = 0;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        for (int j = 0; j < matrix.outerSize(); j++)
            for (IVSparse::SparseMatrix<VALUE_TYPE, int, 3>::InnerIterator it(matrix, j); it; ++it) {
                sum += it.value();
            }
        std::cout << "sum: " << sum;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < matrix.outerSize(); j++)
            for (IVSparse::SparseMatrix<VALUE_TYPE, int, 3>::InnerIterator it(matrix, j); it; ++it) {
                sum += it.value();
            }
        end = std::chrono::high_resolution_clock::now();
        std::cout << "sum: " << sum;
        resultData.at(i).at(6) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
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
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        Eigen::SparseMatrix<VALUE_TYPE> matrix(rows, cols);
        matrix.setFromTriplets(triplet.begin(), triplet.end());
        std::cout << "sum: " << matrix.sum();

    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();

        Eigen::SparseMatrix<VALUE_TYPE> matrix(rows, cols);
        matrix.setFromTriplets(triplet.begin(), triplet.end());

        end = std::chrono::high_resolution_clock::now();
        std::cout << "sum: " << matrix.sum();

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
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = matrix * 2;
        std::cout << result.sum() << std::endl;

    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = matrix * 2;
        end = std::chrono::high_resolution_clock::now();
        std::cout << result.sum() << std::endl;
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
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = eigen_fair_spmv<VALUE_TYPE>(matrix, eigenVector);
        std::cout << result.sum() << std::endl;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = eigen_fair_spmv<VALUE_TYPE>(matrix, eigenVector);
        end = std::chrono::high_resolution_clock::now();
        std::cout << result << std::endl;
        resultData.at(i).at(2) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    std::cout << "sum: " << result.sum() << std::endl;
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
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = eigen_fair_spmm<VALUE_TYPE>(matrix, eigenMatrix);
        std::cout << result.sum() << std::endl;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = eigen_fair_spmm<VALUE_TYPE>(matrix, eigenMatrix);
        end = std::chrono::high_resolution_clock::now();
        std::cout << result.sum() << std::endl;

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
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        sum = matrix.sum();
        std::cout << sum << std::endl;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        sum = matrix.sum();
        end = std::chrono::high_resolution_clock::now();
        std::cout << sum << std::endl;
        resultData.at(i).at(4) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

}


/**
 * @brief Benchmark for Eigen Matrix Transpose
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void eigen_transposeBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols) {
    Eigen::SparseMatrix<VALUE_TYPE> result;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    int sum;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = matrix.transpose().eval();
        assert(result.sum() == matrix.sum());
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        result = matrix.transpose().eval();
        end = std::chrono::high_resolution_clock::now();
        assert(result.sum() == matrix.sum());
        resultData.at(i).at(5) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

}

/**
 * @brief Benchmark for Eigen Matrix iterator traversal
 *
 * @param matrix
 * @param resultData
 * @param numRows
 * @param numCols
 */

void eigen_iteratorBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<uint64_t>>& resultData, int numRows, int numCols) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    int sum = 0;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        for (int j = 0; j < matrix.outerSize(); j++)
            for (Eigen::SparseMatrix<VALUE_TYPE>::InnerIterator it(matrix, j); it; ++it) {
                sum += it.value();
            }
        std::cout << "sum: " << sum << std::endl;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < matrix.outerSize(); j++)
            for (Eigen::SparseMatrix<VALUE_TYPE>::InnerIterator it(matrix, j); it; ++it) {
                sum += it.value();
            }
        end = std::chrono::high_resolution_clock::now();
        std::cout << "sum: " << sum << std::endl;
        resultData.at(i).at(6) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
}


template <typename T, typename indexType, int compressionLevel>
double averageRedundancy(IVSparse::SparseMatrix<T, indexType, compressionLevel>& matrix) {
    const int numRows = matrix.rows();
    const int numCols = matrix.cols();
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
            totalRedundancy += 1;
        else
            redundancy = 1 - (uniqueValues.size() / totalValues);
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
            totalRedundancy += 1;
        else
            redundancy = 1 - (uniqueValues.size() / totalValues);
        totalRedundancy += redundancy;
        colsWithValues++;
    }
    return totalRedundancy / static_cast<double>(colsWithValues);
}


//////////////////////////////
//
//          Due to Eigen's optimizations, we decided to write our own fair spmv and spmm functions. This is purely for creating 
//          an apples to apples comparison between our data structure's implementation and Eigen's. This is a benchmark of a data 
//          structure and not a library.
//
/////////////////////////////
template <typename T, typename indexType, int compressionLevel>
inline Eigen::Matrix<T, -1, 1> IVSparse_fair_spmv(IVSparse::SparseMatrix<T, indexType, compressionLevel>& matrix, Eigen::Matrix<T, -1, 1>& vector) {
    Eigen::Matrix<T, -1, 1> result = Eigen::Matrix<T, -1, 1>::Zero(matrix.rows(), 1);

    for (int j = 0; j < matrix.outerSize(); ++j) {
        for (typename IVSparse::SparseMatrix<T, indexType, compressionLevel>::InnerIterator it(matrix, j); it; ++it) {
            result(it.row()) += it.value() * vector(it.col());
        }
    }

    return result;
}



template <typename T>
inline Eigen::Matrix<T, -1, 1> eigen_fair_spmv(Eigen::SparseMatrix<T>& matrix, Eigen::Matrix<T, -1, 1>& vector) {
    Eigen::Matrix<T, -1, 1> result = Eigen::Matrix<T, -1, 1>::Zero(matrix.rows(), 1);

    for (int j = 0; j < matrix.outerSize(); ++j) {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(matrix, j); it; ++it) {
            result(it.row()) += it.value() * vector(it.col());
        }
    }

    return result;
}

template <typename T>
inline Eigen::Matrix<T, -1, -1> eigen_fair_spmm(Eigen::SparseMatrix<T>& leftMat, Eigen::Matrix<T, -1, -1>& rightMat) {
    Eigen::Matrix<T, -1, -1> result = Eigen::Matrix<T, -1, -1>::Zero(leftMat.rows(), rightMat.cols());

    for (int col = 0; col < rightMat.cols(); col++) {
        for (int row = 0; row < rightMat.rows(); row++) {
            for (typename Eigen::SparseMatrix<T>::InnerIterator matIter(leftMat, row); matIter; ++matIter) {
                result.coeffRef(matIter.row(), col) += matIter.value() * rightMat(row, col);
            }
        }
    }

    return result;
}

template <typename T, typename indexType, int compressionLevel>
inline Eigen::Matrix<T, -1, -1> IVSparse_fair_spmm(IVSparse::SparseMatrix<T, indexType, compressionLevel>& leftMat, Eigen::Matrix<T, -1, -1>& rightMat) {
    Eigen::Matrix<T, -1, -1> result = Eigen::Matrix<T, -1, -1>::Zero(leftMat.rows(), rightMat.cols());

    for (int col = 0; col < rightMat.cols(); col++) {
        for (int row = 0; row < rightMat.rows(); row++) {
            for (typename IVSparse::SparseMatrix<T, indexType, compressionLevel>::InnerIterator matIter(leftMat, row); matIter; ++matIter) {
                result.coeffRef(matIter.row(), col) += matIter.value() * rightMat(row, col);
            }
        }
    }

    return result;
}


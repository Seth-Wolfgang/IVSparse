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
#include <fstream>
#include <cmath>
#include <time.h>

 // General 
#define NUM_ITERATIONS 5
#define NUM_COLD_STARTS 2
#define VALUE_TYPE float
#define TIME_TYPE double

//#define CHECK_VALUES
//#define DEBUG


#define SET_AFFINITY

#ifdef SET_AFFINITY
#include <sched.h>
#endif

// Eigen needs to know size at compile time
#define ROWS 10000
#define COLS 10
#define NNZ 50000

template <typename T, typename indexType, int compressionLevel> double averageRedundancy(IVSparse::SparseMatrix<T, indexType, compressionLevel>& matrix);
template <typename T, typename indexType, int compressionLevel> double averageRedundancy(Eigen::SparseMatrix<T>& matrix);
template <typename T> inline T getMax(std::vector<T> data);
void printDataToFile(std::vector<uint64_t>& data, std::vector<std::vector<TIME_TYPE>>& timeData, const char* filename);

void  VCSC_Benchmark(int, char*);
void IVCSC_Benchmark(int, char*, int);
void eigen_Benchmark(int, char*, int);

void VCSC_outerSumBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData);
void VCSC_CSCConstructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<TIME_TYPE>>& resultData, int rows, int cols);
void VCSC_constructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<TIME_TYPE>>& resultData, int rows, int cols);
void VCSC_scalarBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2> matrix, std::vector<std::vector<TIME_TYPE>>& resultData);
void VCSC_spmvBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numCols);
void VCSC_spmmBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols);
void VCSC_transposeBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols);
void VCSC_iteratorBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols);

void IVCSC_outerSumBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData);
void IVCSC_CSCConstructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<TIME_TYPE>>& resultData, int rows, int cols);
void IVCSC_constructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<TIME_TYPE>>& resultData, int rows, int cols);
void IVCSC_scalarBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3> matrix, std::vector<std::vector<TIME_TYPE>>& resultData);
void IVCSC_spmvBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numCols);
void IVCSC_spmmBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols);
void IVCSC_transposeBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols);
void IVCSC_iteratorBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols);

void eigen_outerSumBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData);
void eigen_constructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<TIME_TYPE>>& resultData, int rows, int cols);
void eigen_scalarBenchmark(Eigen::SparseMatrix<VALUE_TYPE> matrix, std::vector<std::vector<TIME_TYPE>>& resultData);
void eigen_spmvBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numCols);
void eigen_spmmBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols);
void eigen_transposeBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols);
void eigen_iteratorBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols);

template <typename T> inline Eigen::Matrix<T, -1, -1> eigen_fair_spmm(Eigen::SparseMatrix<T>& leftMat, Eigen::Matrix<T, -1, -1>& rightMat);
template <typename T> inline Eigen::Matrix<T, -1, 1> eigen_fair_spmv(Eigen::SparseMatrix<T>& matrix, Eigen::Matrix<T, -1, 1>& vector);
template <typename T, typename indexType, int compressionLevel> inline Eigen::Matrix<T, -1, -1> IVSparse_fair_spmm(IVSparse::SparseMatrix<T, indexType, compressionLevel>& leftMat, Eigen::Matrix<T, -1, -1>& rightMat);
template <typename T, typename indexType, int compressionLevel> inline Eigen::Matrix<T, -1, 1> IVSparse_fair_spmv(IVSparse::SparseMatrix<T, indexType, compressionLevel>& matrix, Eigen::Matrix<T, -1, 1>& vector);

void readCSC(const char* valsPath, const char* innerPath, const char* outerPath);

// Global values for checking sums
volatile VALUE_TYPE VCSC_ScalarSum;
volatile VALUE_TYPE VCSC_SpmvSum;
volatile VALUE_TYPE VCSC_SpmmSum;
volatile VALUE_TYPE VCSC_ConstructorSum;
volatile VALUE_TYPE VCSC_IteratorSum;
volatile VALUE_TYPE VCSC_TransposeSum;

volatile VALUE_TYPE IVCSC_ScalarSum;
volatile VALUE_TYPE IVCSC_SpmvSum;
volatile VALUE_TYPE IVCSC_SpmmSum;
volatile VALUE_TYPE IVCSC_ConstructorSum;
volatile VALUE_TYPE IVCSC_IteratorSum;
volatile VALUE_TYPE IVCSC_TransposeSum;

volatile VALUE_TYPE Eigen_ScalarSum;
volatile VALUE_TYPE Eigen_SpmvSum;
volatile VALUE_TYPE Eigen_SpmmSum;
volatile VALUE_TYPE Eigen_ConstructorSum;
volatile VALUE_TYPE Eigen_IteratorSum;
volatile VALUE_TYPE Eigen_TransposeSum;

Eigen::Matrix<VALUE_TYPE, -1, -1> eigenMatrix;
Eigen::Matrix<VALUE_TYPE, -1, 1> eigenVector;

int id;
double redundancy;

std::vector<std::tuple<int, int, VALUE_TYPE>> data;


int main(int argc, char** argv) {
    char* vals = argv[1];
    char* innerPath = argv[2];
    char* outerPath = argv[3];
	redundancy = atof(argv[4]);
    id = atoi(argv[5]);
    int which = atoi(argv[6]);
	char* resultsPath = argv[7];
    srand(1);
    std::cout << "Rows: " << ROWS << " Cols: " << COLS << " NNZ: " << NNZ << " Redundancy: " << redundancy << "\n";
    readCSC(vals, innerPath, outerPath);

    eigenMatrix = Eigen::Matrix<VALUE_TYPE, -1, -1>::Random(COLS, 1000);
    eigenVector = Eigen::Matrix<VALUE_TYPE, -1, 1>::Random(COLS, 1);

    VCSC_ScalarSum = 0;
    VCSC_SpmvSum = 0;
    VCSC_SpmmSum = 0;
    VCSC_ConstructorSum = 0;
    VCSC_IteratorSum = 0;
    VCSC_TransposeSum = 0;

    IVCSC_ScalarSum = 0;
    IVCSC_SpmvSum = 0;
    IVCSC_SpmmSum = 0;
    IVCSC_ConstructorSum = 0;
    IVCSC_IteratorSum = 0;
    IVCSC_TransposeSum = 0;

    Eigen_ScalarSum = 0;
    Eigen_SpmvSum = 0;
    Eigen_SpmmSum = 0;
    Eigen_ConstructorSum = 0;
    Eigen_IteratorSum = 0;
    Eigen_TransposeSum = 0;

    std::cout << "which: " << which << "\n";

	//std::ios_base::sync_with_stdio(false); // modded
    if (which == -1 || which == 0) {
		std::cout << "starting with vcsc\n";
        VCSC_Benchmark(0, resultsPath);
		std::cout << "done with vcsc\n";
    }
	if (which == -1 || which == 1) {
		std::cout << "starting with ivcsc\n";
		IVCSC_Benchmark(2, resultsPath, which);
		std::cout << "done with ivcsc\n";
    }
    if (which == -1 || which == 2) {
		std::cout << "starting with eigen\n";
        eigen_Benchmark(4, resultsPath, which);
		std::cout << "done with eigen\n";
    }

	std::cout << "VCSC_ScalarSum: " << VCSC_ScalarSum << "\n";
	std::cout << "VCSC_SpmvSum: " << VCSC_SpmvSum << "\n";
	std::cout << "VCSC_SpmmSum: " << VCSC_SpmmSum << "\n";
	std::cout << "VCSC_ConstructorSum: " << VCSC_ConstructorSum << "\n";
	std::cout << "VCSC_IteratorSum: " << VCSC_IteratorSum << "\n";
	std::cout << "VCSC_TransposeSum: " << VCSC_TransposeSum << "\n";
	
	std::cout << "IVCSC_ScalarSum: " << IVCSC_ScalarSum << "\n";
	std::cout << "IVCSC_SpmvSum: " << IVCSC_SpmvSum << "\n";
	std::cout << "IVCSC_SpmmSum: " << IVCSC_SpmmSum << "\n";
	std::cout << "IVCSC_ConstructorSum: " << IVCSC_ConstructorSum << "\n";
	std::cout << "IVCSC_IteratorSum: " << IVCSC_IteratorSum << "\n";
	std::cout << "IVCSC_TransposeSum: " << IVCSC_TransposeSum << "\n";
	
	std::cout << "Eigen_ScalarSum: " << Eigen_ScalarSum << "\n";
	std::cout << "Eigen_SpmvSum: " << Eigen_SpmvSum << "\n";
	std::cout << "Eigen_SpmmSum: " << Eigen_SpmmSum << "\n";
	std::cout << "Eigen_ConstructorSum: " << Eigen_ConstructorSum << "\n";
	std::cout << "Eigen_IteratorSum: " << Eigen_IteratorSum << "\n";
	std::cout << "Eigen_TransposeSum: " << Eigen_TransposeSum << "\n";
    
	std::cout << "all finished\n";
    return 0;
}

/*************************************************************************************************************
 *                                                                                                           *
 *                                                                                                           *
 *                                           Helper Functions                                                *
 *                                                                                                           *
 *                                                                                                           *
 *                                                                                                           *
 ************************************************************************************************************/

void readCSC(const char* valsPath, const char* innerPath, const char* outerPath) {

    
    data.reserve(NNZ);
        
    std::ifstream valsFile(valsPath);
    std::ifstream innerFile(innerPath);
    std::ifstream outerFile(outerPath);
	

    int idx = 0; 
    int cur_col_start_idx = 0; 
    int next_col_start_idx = 0;
    int curr_row = 0;

    VALUE_TYPE p = 0.;
    outerFile >> cur_col_start_idx;
	outerFile >> next_col_start_idx;
	//std::cout << "here\n";
	//std::cout << j << " " << q << "\n";
    for (int i = 0; i < COLS; ++i) {
        for (int j=cur_col_start_idx; j < next_col_start_idx; ++j, idx++) {
            valsFile >> p;
            innerFile >> curr_row;
            data.emplace_back(curr_row, i, p);
			//std::cout << curr_row << " " << i << " " << p << "\n";
		}
        cur_col_start_idx = next_col_start_idx;
		outerFile >> next_col_start_idx;
    }
    
    //std::cout << "done reading in matrix\n";
    //std::cout << "num vals read in: " << idx << "\n";
    valsFile.close();
    innerFile.close();
    outerFile.close();


}


/**
 * @brief gets the max in the tuple
 *
 *
 */

template <typename T>
inline T getMax(std::vector<T> data) {
    T max = 0;
    for (int i = 0; i < data.size(); i++) {
        if (data.at(i) > max) {
            max = data.at(i);
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

void printDataToFile(std::vector<double>& data, std::vector<std::vector<TIME_TYPE>>& timeData, const char* filename) {

	FILE* file;

    //check if file exists
    int fileExists = access(filename, F_OK);
    if (fileExists == -1) {
        file = fopen(filename, "a");
		if (file == NULL) {
			std::cerr << "FAILED TO OPEN RESULTS FILE -- QUITTING\n";
			exit(-1);
		}
        fprintf(file, "%s\n", "ID,rows,cols,nonzeros,sparsity,redundancy,size,constructor_time,scalar_time,spmv_time,spmm_time,sum_time,transpose_time,iterator_time");
        fclose(file);
    }
    file = fopen(filename, "a");

    // for (uint64_t i = 0; i < timeData[i].size(); i++) {
    //     std::cout << data.at(4) << " ";
    //     std::cout << timeData.at(i).at(0) << " " << timeData.at(i).at(1) << " " << timeData.at(i).at(2) << " " <<
    //         timeData.at(i).at(3) << " " << timeData.at(i).at(4) << " " << timeData.at(i).at(5) << " " << timeData.at(i).at(6) << "\n";
    // }

    /**
     * ID, rows, cols, nonzeros, sparsity, redundancy, size,
     *
     * scalar time, spmv time, spmm time
     */
     // double redundancy = (double)(1.0 / data.at(3));

    for (uint64_t i = 0; i < timeData.size(); i++) {
        fprintf(file, "%.0lf, %.0lf, %.0lf, %lf, %lf, %lf, %lf, %.16f, %.16f, %.16f, %.16f, %.16f, %.16f, %.16f\n",
                data.at(0),
                data.at(1),
                data.at(2),
                data.at(3),
                0.0,
                redundancy,
                data.at(5),
                timeData.at(i).at(0),
                timeData.at(i).at(1),
                timeData.at(i).at(2),
                timeData.at(i).at(3),
                timeData.at(i).at(4),
                timeData.at(i).at(5),
                timeData.at(i).at(6));

    }
    fclose(file);
}

/**
 * @brief function to help control all VCSC benchmarks
 *
 * @param data
 * @param numRows
 * @param numCol
 * @param nonzeros
 */

void help_affinity(int aff) {
    #ifdef SET_AFFINITY
    cpu_set_t  mask;
    CPU_ZERO(&mask);
    CPU_SET(aff, &mask);
    int result = sched_setaffinity(0, sizeof(mask), &mask);
    #endif
}

void  VCSC_Benchmark(int aff, char *path) {
    
    help_affinity(aff);

    std::vector<double> matrixData(1);
    std::vector<std::vector<TIME_TYPE>> timeData(NUM_ITERATIONS);

    IVSparse::SparseMatrix<VALUE_TYPE, int, 2> matrix(data, ROWS, COLS, data.size());

    matrixData.resize(6);
    matrixData.at(0) = id;
    matrixData.at(1) = ROWS;
    matrixData.at(2) = COLS;
    matrixData.at(3) = NNZ;
    matrixData.at(4) = redundancy;
    matrixData.at(5) = matrix.byteSize();

    for (int j = 0; j < NUM_ITERATIONS; j++) {
        timeData.at(j).resize(7);
    }

    // matrix.print();

    VCSC_constructorBenchmark(data, timeData, ROWS, COLS);
	#ifdef DEBUG
		std::cout << "VCSC constructor done" << "\n";
	#endif

	VCSC_scalarBenchmark(matrix, timeData);
	#ifdef DEBUG
		std::cout << "VCSC scalar done" << "\n";
	#endif

	// VCSC_outerSumBenchmark(matrix, timeData);
    // std::cout << "VCSC column sums done" << "\n";
	VCSC_spmvBenchmark(matrix, timeData, COLS);
	#ifdef DEBUG
		std::cout << "VCSC spmv done" << "\n";
	#endif
	
	VCSC_spmmBenchmark(matrix, timeData, ROWS, COLS);
	#ifdef DEBUG
	std::cout << "VCSC spmm done" << "\n";
	#endif
	
	VCSC_iteratorBenchmark(matrix, timeData, ROWS, COLS);
	#ifdef DEBUG
	std::cout << "VCSC iterator done\n" << "\n";
	#endif
	
	VCSC_transposeBenchmark(matrix, timeData, ROWS, COLS);
	#ifdef DEBUG
	std::cout << "VCSC transpose done\n" << "\n";
	#endif

	/*
    std::stringstream path;
    path << "../results/VCSCResults_COO.csv";
	printDataToFile(matrixData, timeData, path.str().c_str());
	*/
	std::string fullpath = "vcsc-results.csv";
	fullpath = path + fullpath;
	printDataToFile(matrixData, timeData, fullpath.c_str());
    // adjustValues(data, static_cast<int>(((double)numRows / (double)MATRICES) * (i + 1)), i);

}

/**
 * @brief Function to help control all IVCSC benchmarks
 *
 * @param data
 * @param numRows
 * @param numCol
 * @param nonzeros
 */

void   IVCSC_Benchmark(int aff, char *path, int which) {
    
    help_affinity(aff);

    std::vector<double> matrixData(1);
    std::vector<std::vector<TIME_TYPE>> timeData(NUM_ITERATIONS);
    // adjustValues(data, 1, 1);

    IVSparse::SparseMatrix<VALUE_TYPE, int, 3> matrix(data, ROWS, COLS, data.size());

    matrixData.resize(6);
    matrixData.at(0) = id;
    matrixData.at(1) = ROWS;
    matrixData.at(2) = COLS;
    matrixData.at(3) = NNZ;
    matrixData.at(4) = redundancy;
    matrixData.at(5) = matrix.byteSize();

    for (int j = 0; j < NUM_ITERATIONS; j++) {
        timeData.at(j).resize(7);
    }

    IVCSC_constructorBenchmark(data, timeData, ROWS, COLS);
	#ifdef DEBUG
		std::cout << "IVCSC constructor done" << "\n";
	#endif

	IVCSC_scalarBenchmark(matrix, timeData);
	#ifdef DEBUG
	std::cout << "IVCSC scalar done" << "\n";
    #endif

	// IVCSC_outerSumBenchmark(matrix, timeData);
    // std::cout << "IVCSC column sums done" << "\n";
    
	IVCSC_spmvBenchmark(matrix, timeData, COLS);
	#ifdef DEBUG
		std::cout << "IVCSC spmv done" << "\n";
	#endif

	IVCSC_spmmBenchmark(matrix, timeData, ROWS, COLS);
	#ifdef DEBUG
	std::cout << "IVCSC spmm done" << "\n";
	#endif

	IVCSC_iteratorBenchmark(matrix, timeData, ROWS, COLS);
	#ifdef DEBUG
	std::cout << "IVCSC iterator done" << "\n";
	#endif
	
	IVCSC_transposeBenchmark(matrix, timeData, ROWS, COLS);
	#ifdef DEBUG
	std::cout << "IVCSC transpose done" << "\n";
	#endif

    #ifdef CHECK_VALUES
    if (which == -1) {
        assert(0.1 > std::abs(IVCSC_ScalarSum - VCSC_ScalarSum));
        assert(0.1 > std::abs(IVCSC_SpmvSum - VCSC_SpmvSum));
        assert(0.1 > std::abs(IVCSC_SpmmSum - VCSC_SpmmSum));
        assert(0.1 > std::abs(IVCSC_ConstructorSum - VCSC_ConstructorSum));
        assert(0.1 > std::abs(IVCSC_IteratorSum - VCSC_IteratorSum));
        assert(0.1 > std::abs(IVCSC_TransposeSum - VCSC_TransposeSum));
    }
    #endif

	/*
    std::stringstream path;
    path << "../results/IVCSCResults_COO.csv";
    printDataToFile(matrixData, timeData, path.str().c_str());
	*/
	std::string fullpath = "ivcsc_results.csv";
	fullpath = path + fullpath;
	printDataToFile(matrixData, timeData, fullpath.c_str());
    // adjustValues(data, static_cast<int>(((double)numRows / (double)MATRICES) * (i + 1)), i);


}

/**
 * @brief function to help cntrol all eigen benchmarks
 *
 * @param data
 * @param numRows
 * @param numCol
 * @param nonzeros
 */

void eigen_Benchmark(int aff, char *path, int which) {
    
    help_affinity(aff);

    Eigen::SparseMatrix<VALUE_TYPE> matrix(ROWS, COLS);
    matrix.reserve(data.size());
    std::vector<Eigen::Triplet<VALUE_TYPE>> triplet;
    std::vector<double> matrixData(1);
    std::vector<std::vector<TIME_TYPE>> timeData(NUM_ITERATIONS);

    triplet.clear();
    triplet.reserve(data.size());
    for (int j = 0; j < data.size(); j++) {
        triplet.push_back(Eigen::Triplet<VALUE_TYPE>(std::get<0>(data.at(j)), std::get<1>(data.at(j)), std::get<2>(data.at(j))));
    }
    matrix.setFromTriplets(triplet.begin(), triplet.end());


    matrixData.resize(6);
    matrixData.at(0) = id;
    matrixData.at(1) = ROWS;
    matrixData.at(2) = COLS;
    matrixData.at(3) = NNZ;
    matrixData.at(4) = redundancy;
    matrixData.at(5) = matrix.nonZeros() * sizeof(VALUE_TYPE) + matrix.nonZeros() * sizeof(uint32_t) + (matrix.outerSize() + 1) * sizeof(uint32_t);

    for (int k = 0; k < NUM_ITERATIONS; k++) {
        timeData.at(k).resize(7);
    }

    eigen_constructorBenchmark(data, timeData, ROWS, COLS);
	#ifdef DEBUG
	std::cout << "Eigen constructor done" << "\n";
	#endif

	eigen_scalarBenchmark(matrix, timeData);
	#ifdef DEBUG
	std::cout << "Eigen scalar done" << "\n";
	#endif

	// eigen_outerSumBenchmark(matrix, timeData);
    // std::cout << "Eigen column sums done" << "\n";
	eigen_spmvBenchmark(matrix, timeData, COLS);
	#ifdef DEBUG
	std::cout << "Eigen spmv done" << "\n";
	#endif
		
	eigen_spmmBenchmark(matrix, timeData, ROWS, COLS);
	#ifdef DEBUG
	std::cout << "Eigen spmm done" << "\n";
	#endif
	
	eigen_iteratorBenchmark(matrix, timeData, ROWS, COLS);
	#ifdef DEBUG
	std::cout << "Eigen iterator done" << "\n";
	#endif
	
	eigen_transposeBenchmark(matrix, timeData, ROWS, COLS);
	#ifdef DEBUG
	std::cout << "Eigen transpose done" << "\n";
	#endif

    #ifdef CHECK_VALUES
    if (which == -1) {
        assert(0.1 > std::abs(Eigen_ScalarSum - VCSC_ScalarSum));
        assert(0.1 > std::abs(Eigen_SpmvSum - VCSC_SpmvSum));
        assert(0.1 > std::abs(Eigen_SpmmSum - VCSC_SpmmSum));
        assert(0.1 > std::abs(Eigen_ConstructorSum - VCSC_ConstructorSum));
        assert(0.1 > std::abs(Eigen_IteratorSum - VCSC_IteratorSum));
        assert(0.1 > std::abs(Eigen_TransposeSum - VCSC_TransposeSum));
    }
    #endif

	/*
    std::stringstream path;
    path << "../results/EigenResults_COO.csv";
	printDataToFile(matrixData, timeData, path.str().c_str());
	*/
	std::string fullpath = "eigen_results.csv";
	fullpath = path + fullpath;
	printDataToFile(matrixData, timeData, fullpath.c_str());

    // adjustValues(data, static_cast<int>(((double)numRows / (double)MATRICES) * (i + 1)), i);

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

void   VCSC_constructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<TIME_TYPE>>& resultData, int rows, int cols) {
    clock_t start, end;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        IVSparse::SparseMatrix<VALUE_TYPE, int, 2> matrix(data, rows, cols, data.size());
        VCSC_ConstructorSum += matrix.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = clock();
        IVSparse::SparseMatrix<VALUE_TYPE, int, 2> matrix(data, rows, cols, data.size());
        end = clock();
		
		VCSC_ConstructorSum += matrix.sum();
		resultData.at(i).at(0) = (double) (end-start)/(double)CLOCKS_PER_SEC;
		std::cout << end << "," << start << "," << CLOCKS_PER_SEC << "\n";
		#ifdef DEBUG
        std::cout << VCSC_ConstructorSum << "\n";
		#endif
    }
}

/**
 * @brief Benchmark for VCSC scalar multiplication
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void   VCSC_scalarBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2> matrix, std::vector<std::vector<TIME_TYPE>>& resultData) {
    clock_t start, end;

    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        matrix *= 2;
		VCSC_ScalarSum += matrix.sum();
    }

    start = clock();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        matrix *= 2;
    }
    end = clock();
	VCSC_ScalarSum += matrix.sum();
	#ifdef DEBUG
    std::cout << "sum: " << VCSC_ScalarSum << "\n";
	#endif
    resultData.at(NUM_ITERATIONS-1).at(1) = (double) (end-start)/(double)CLOCKS_PER_SEC;

}

/**
 * @brief Benchmark for VCSC Matrix * (dense) vector multiplication
 *
 * @param matrix
 * @param numCols
 */

void   VCSC_spmvBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numCols) {
    clock_t start, end;
    Eigen::Matrix<VALUE_TYPE, -1, 1> result;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = IVSparse_fair_spmv<VALUE_TYPE, int, 2>(matrix, eigenVector);
		VCSC_SpmvSum += result.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        result = IVSparse_fair_spmv<VALUE_TYPE, int, 2>(matrix, eigenVector);
		end = clock();
		VCSC_SpmvSum += result.sum();
		#ifdef DEBUG
        std::cout << "sum: " << VCSC_SpmvSum << "\n";
		#endif
		resultData.at(i).at(2) = (double) (end-start)/(double)CLOCKS_PER_SEC;
    }

}

/**
 * @brief Benchmark for VCSC Matrix * Matrix multiplication
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void VCSC_spmmBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols) {
    Eigen::Matrix<VALUE_TYPE, -1, -1> result;
    clock_t start, end;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = IVSparse_fair_spmm<VALUE_TYPE, int, 2>(matrix, eigenMatrix);
		VCSC_SpmmSum += result.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        result = IVSparse_fair_spmm<VALUE_TYPE, int, 2>(matrix, eigenMatrix);
		end = clock();
		VCSC_SpmmSum += result.sum();
		#ifdef DEBUG
        std::cout << "sum: " << VCSC_SpmmSum << "\n";
		#endif
		resultData.at(i).at(3) = (double) (end-start)/(double)CLOCKS_PER_SEC;
    }

}

/**
 * @brief Benchmark for column sums
 *
 * @param matrix
 * @param resultData
 */

void VCSC_outerSumBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData) {
    clock_t start, end;
    IVSparse::SparseMatrix<VALUE_TYPE, int, 2> result;
    int sum;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        sum = matrix.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        sum = matrix.sum();
		end = clock();
		exit(-1);
		#ifdef DEBUG
		std::cout << "sum: " << sum;
		#endif
		resultData.at(i).at(4) = (double) (end-start)/(double)CLOCKS_PER_SEC;
    }
}

/**
 * @brief Benchmark for VCSC Matrix Transpose
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void VCSC_transposeBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols) {
    clock_t start, end;
    IVSparse::SparseMatrix<VALUE_TYPE, int, 2> result;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = matrix.transpose();
		VCSC_TransposeSum += result.sum();
		#ifdef CHECK_VALUES
		assert(std::abs(result.sum() -  matrix.sum()) < 1e-8);
		#endif
	}

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        result = matrix.transpose();
		end = clock();
		VCSC_TransposeSum += result.sum();
		#ifdef DEBUG
        std::cout << "sum: " << VCSC_TransposeSum << "\n";
		#endif
		#ifdef CHECK_VALUES
		assert(std::abs(result.sum() -  matrix.sum()) < 1e-8);
		#endif
		resultData.at(i).at(5) = (double) (end-start)/(double)CLOCKS_PER_SEC;
    }

}

/**
 * @brief Benchmark for VCSC Matrix iterator traversal
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void VCSC_iteratorBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 2>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols) {
    clock_t start, end;
    double sum = 0;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        for (int j = 0; j < matrix.outerSize(); j++)
            for (IVSparse::SparseMatrix<VALUE_TYPE, int, 2>::InnerIterator it(matrix, j); it; ++it) {
                sum += it.value();
            }
		VCSC_IteratorSum += sum;
		sum = 0;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        for (int j = 0; j < matrix.outerSize(); j++)
            for (IVSparse::SparseMatrix<VALUE_TYPE, int, 2>::InnerIterator it(matrix, j); it; ++it) {
                sum += it.value();
            }

		end = clock();
		VCSC_IteratorSum += sum;
		sum = 0;
		#ifdef DEBUG
        std::cout << "sum: " << sum << "\n";
		#endif
		resultData.at(i).at(6) = (double) (end-start)/(double)CLOCKS_PER_SEC;
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

void IVCSC_constructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<TIME_TYPE>>& resultData, int rows, int cols) {
    clock_t start, end;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        IVSparse::SparseMatrix<VALUE_TYPE, int, 3> matrix(data, rows, cols, data.size());
        IVCSC_ConstructorSum += matrix.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        IVSparse::SparseMatrix<VALUE_TYPE, int, 3> matrix(data, rows, cols, data.size());
		end = clock();
        IVCSC_ConstructorSum += matrix.sum();
		#ifdef DEBUG
        std::cout << "sum: " << IVCSC_ConstructorSum << "\n";
		#endif
		resultData.at(i).at(0) = (double) (end-start)/(double)CLOCKS_PER_SEC;
    }
}

/**
 * @brief Benchmark for IVCSC scalar multiplication
 *
 * @param matrix
 */

void IVCSC_scalarBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3> matrix, std::vector<std::vector<TIME_TYPE>>& resultData) {
    clock_t start, end;

    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        matrix *= 2;
		IVCSC_ScalarSum += matrix.sum();
    }

	start = clock();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        matrix *= 2;
    }
	end = clock();
	IVCSC_ScalarSum += matrix.sum();
	#ifdef DEBUG
    std::cout << "sum: " << IVCSC_ScalarSum << "\n";
	#endif
	resultData.at(NUM_ITERATIONS-1).at(1) = (double) (end-start)/(double)CLOCKS_PER_SEC;

}

/**
 * @brief Benchmark for Sparse Matrix * (dense) Vector
 *
 * @param matrix
 * @param resultData
 * @param numCols
 */

void   IVCSC_spmvBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numCols) {
    clock_t start, end;
    Eigen::Matrix<VALUE_TYPE, -1, 1> result;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = IVSparse_fair_spmv<VALUE_TYPE, int, 3>(matrix, eigenVector);
		IVCSC_SpmvSum += result.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        result = IVSparse_fair_spmv<VALUE_TYPE, int, 3>(matrix, eigenVector);
		end = clock();
		IVCSC_SpmvSum += result.sum();
		#ifdef DEBUG
		std::cout << "sum: " << IVCSC_SpmvSum << "\n";
		#endif
		resultData.at(i).at(2) = (double) (end-start)/(double)CLOCKS_PER_SEC;
    }
}


/**
 * @brief Benchmark for IVCSC Matrix * (dense) Matrix multiplication
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void IVCSC_spmmBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols) {
    Eigen::Matrix<VALUE_TYPE, -1, -1> result;
    clock_t start, end;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = IVSparse_fair_spmm<VALUE_TYPE, int, 3>(matrix, eigenMatrix);
		IVCSC_SpmmSum += result.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        result = IVSparse_fair_spmm<VALUE_TYPE, int, 3>(matrix, eigenMatrix);
		end = clock();
		IVCSC_SpmmSum += result.sum();
		#ifdef DEBUG
        std::cout << "sum: " << IVCSC_SpmmSum << "\n";
		#endif
		resultData.at(i).at(3) = (double) (end-start)/(double)CLOCKS_PER_SEC;
    }

}

/**
 * @brief  Benchmark for IVCSC Matrix column sums
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */


void IVCSC_outerSumBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData) {
    clock_t start, end;
    IVSparse::SparseMatrix<VALUE_TYPE, int, 3> result;
    int sum;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        sum = matrix.sum();
    }


    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        sum = matrix.sum();
		end = clock();
		exit(-1);
		#ifdef DEBUG
        std::cout << "sum: " << sum;
		#endif
		resultData.at(i).at(4) = (double) (end-start)/(double)CLOCKS_PER_SEC;
    }

}



/**
 * @brief Benchmark for IVCSC Matrix Transpose
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void IVCSC_transposeBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols) {
    clock_t start, end;
    IVSparse::SparseMatrix<VALUE_TYPE, int, 3> result;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = matrix.transpose();
		IVCSC_TransposeSum += result.sum();
		#ifdef CHECK_VALUES
		assert(std::abs(result.sum() -  matrix.sum()) < 1e-8);
		#endif
	}

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        result = matrix.transpose();
		end = clock();
		IVCSC_TransposeSum += result.sum();
		#ifdef DEBUG
        std::cout << "sum: " << IVCSC_TransposeSum << "\n";
		#endif	
		#ifdef CHECK_VALUES
		assert(std::abs(result.sum() -  matrix.sum()) < 1e-8);
		#endif
		resultData.at(i).at(5) = (double) (end-start)/(double)CLOCKS_PER_SEC;
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

void IVCSC_iteratorBenchmark(IVSparse::SparseMatrix<VALUE_TYPE, int, 3>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols) {
    clock_t start, end;
    double sum = 0;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        for (int j = 0; j < matrix.outerSize(); j++)
            for (IVSparse::SparseMatrix<VALUE_TYPE, int, 3>::InnerIterator it(matrix, j); it; ++it) {
                sum += it.value();
            }
		IVCSC_IteratorSum += sum;
		sum = 0;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        for (int j = 0; j < matrix.outerSize(); j++)
            for (IVSparse::SparseMatrix<VALUE_TYPE, int, 3>::InnerIterator it(matrix, j); it; ++it) {
                sum += it.value();
            }
		end = clock();
		IVCSC_IteratorSum += sum;
		sum = 0;
		#ifdef DEBUG
        std::cout << "sum: " << sum << "\n";
		#endif
		resultData.at(i).at(6) = (double) (end-start)/(double)CLOCKS_PER_SEC;
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

void   eigen_constructorBenchmark(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, std::vector<std::vector<TIME_TYPE>>& resultData, int rows, int cols) {
    clock_t start, end;
    Eigen::SparseMatrix<VALUE_TYPE> matrix(rows, cols);
    std::vector<Eigen::Triplet<VALUE_TYPE>> triplet;

    for (int i = 0; i < data.size(); i++) {
        triplet.push_back(Eigen::Triplet<VALUE_TYPE>(std::get<0>(data.at(i)), std::get<1>(data.at(i)), std::get<2>(data.at(i))));
    }

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        Eigen::SparseMatrix<VALUE_TYPE> matrix(rows, cols);
        matrix.setFromTriplets(triplet.begin(), triplet.end());
        Eigen_ConstructorSum += matrix.sum();

    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();

        Eigen::SparseMatrix<VALUE_TYPE> matrix(rows, cols);
        matrix.setFromTriplets(triplet.begin(), triplet.end());
		end = clock();
		Eigen_ConstructorSum += matrix.sum();
		#ifdef DEBUG
        std::cout << "sum: " << Eigen_ConstructorSum << "\n";
		#endif
		resultData.at(i).at(0) = (double) (end-start)/(double)CLOCKS_PER_SEC;
    }
}

/**
 * @brief Benchmark for Eigen scalar multiplication
 *
 * @param matrix
 */

void   eigen_scalarBenchmark(Eigen::SparseMatrix<VALUE_TYPE> matrix, std::vector<std::vector<TIME_TYPE>>& resultData) {
    clock_t start, end;

    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        matrix *= 2;
		Eigen_ScalarSum += matrix.sum();
    }

	start = clock();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        matrix *= 2;
    }
	end = clock();
	Eigen_ScalarSum += matrix.sum();
	#ifdef DEBUG
    std::cout << "sum: " << Eigen_ScalarSum << "\n";
	#endif
	resultData.at(NUM_ITERATIONS-1).at(1) = (double) (end-start)/(double)CLOCKS_PER_SEC;

}

/**
 * @brief Benchmark for Eigen Matrix * vector multiplication
 *
 * @param matrix
 * @param numRows
 */

void eigen_spmvBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numCols) {
    Eigen::Matrix<VALUE_TYPE, -1, 1> result;
    clock_t start, end;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = eigen_fair_spmv<VALUE_TYPE>(matrix, eigenVector);
		Eigen_SpmvSum += result.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        result = eigen_fair_spmv<VALUE_TYPE>(matrix, eigenVector);
		end = clock();
		Eigen_ConstructorSum += matrix.sum();
		Eigen_SpmvSum += result.sum();
		#ifdef DEBUG
		std::cout << "sum: " << EigenSpmvSum << "\n";
		#endif
		resultData.at(i).at(2) = (double) (end-start)/(double)CLOCKS_PER_SEC;
    }
}




/**
 * @brief Benchmark for Eigen Matrix * Matrix multiplication
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void eigen_spmmBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols) {
    Eigen::Matrix<VALUE_TYPE, -1, -1> result;
    clock_t start, end;
    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = eigen_fair_spmm<VALUE_TYPE>(matrix, eigenMatrix);
		Eigen_SpmmSum += result.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        result = eigen_fair_spmm<VALUE_TYPE>(matrix, eigenMatrix);
		end = clock();
		Eigen_SpmmSum += result.sum();
		#ifdef DEBUG
        std::cout << Eigen_SpmmmSum << "\n";
		#endif
		resultData.at(i).at(3) = (double) (end-start)/(double)CLOCKS_PER_SEC;
    }
}

/**
 * @brief Benchmark for column sums
 *
 * @param matrix
 * @param resultData
 */

void eigen_outerSumBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData) {
    Eigen::SparseMatrix<VALUE_TYPE> result;
    clock_t start, end;
    int sum;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        sum = matrix.sum();
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        sum = matrix.sum();
		end = clock();
		exit(-1);
		#ifdef DEBUG
		std::cout << sum << "\n";
		#endif
		resultData.at(i).at(4) = (double) (end-start)/(double)CLOCKS_PER_SEC;
    }

}


/**
 * @brief Benchmark for Eigen Matrix Transpose
 *
 * @param matrix
 * @param numRows
 * @param numCols
 */

void eigen_transposeBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols) {
    Eigen::SparseMatrix<VALUE_TYPE> result;
    clock_t start, end;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        result = matrix.transpose().eval();
		Eigen_TransposeSum += result.sum();
		#ifdef CHECK_VALUES
		assert(std::abs(result.sum() -  matrix.sum()) < 1e-8);
		#endif
	}

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        result = matrix.transpose().eval();
		end = clock();
		Eigen_TransposeSum += result.sum();
		#ifdef DEBUG
        std::cout << "sum: " << Eigen_TransposeSum << "\n";
		#endif
		#ifdef CHECK_VALUES
		assert(std::abs(result.sum() -  matrix.sum()) < 1e-8);
		#endif
		resultData.at(i).at(5) = (double) (end-start)/(double)CLOCKS_PER_SEC;
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

void eigen_iteratorBenchmark(Eigen::SparseMatrix<VALUE_TYPE>& matrix, std::vector<std::vector<TIME_TYPE>>& resultData, int numRows, int numCols) {
    clock_t start, end;
    double sum = 0;

    //cold start
    for (int i = 0; i < NUM_COLD_STARTS; i++) {
        for (int j = 0; j < matrix.outerSize(); j++)
            for (Eigen::SparseMatrix<VALUE_TYPE>::InnerIterator it(matrix, j); it; ++it) {
                sum += it.value();
            }
		Eigen_IteratorSum += sum;
		sum = 0;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
		start = clock();
        for (int j = 0; j < matrix.outerSize(); j++)
            for (Eigen::SparseMatrix<VALUE_TYPE>::InnerIterator it(matrix, j); it; ++it) {
                sum += it.value();
            }
		end = clock();
		Eigen_IteratorSum += sum;
		sum = 0;
		#ifdef DEBUG
		std::cout << "sum: " << sum << "\n";
		#endif
		resultData.at(i).at(6) = (double) (end-start)/(double)CLOCKS_PER_SEC;
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
    std::cout << totalRedundancy << "\n";
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
            result(it.row()) += it.value() * vector(j);
        }
    }

    return result;
}


template <typename T>
inline Eigen::Matrix<T, -1, 1> eigen_fair_spmv(Eigen::SparseMatrix<T>& matrix, Eigen::Matrix<T, -1, 1>& vector) {
    Eigen::Matrix<T, -1, 1> result = Eigen::Matrix<T, -1, 1>::Zero(matrix.rows(), 1);

    for (int j = 0; j < matrix.outerSize(); ++j) {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(matrix, j); it; ++it) {
            result(it.row()) += it.value() * vector(j);
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



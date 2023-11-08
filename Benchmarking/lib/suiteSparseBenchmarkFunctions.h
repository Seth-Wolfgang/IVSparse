/**
 * @file benchmarkFunctions.h
 * @author your name (you@domain.com)
 * @brief For function definitions in benchmark.cpp
 * @version 0.1
 * @date 2023-08-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */


#include <chrono>
#include "../../IVSparse/SparseMatrix"
#include <unordered_set>
#include "suiteSparseBenchmarkAnalysis.cpp"
#include "mmio.c"
#include "armadillo" // https://arma.sourceforge.net/
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iterator>
#include <iostream>
#include <string>

#define VALUE_TYPE int
#define INDEX_TYPE int
#define NUM_OF_BENCHMARKS 50

// Function to read Matrix Market files
template <typename T>
void readFile(std::vector<Eigen::Triplet<T>> &eigenTriplet, std::vector<double> &matrixData, char *filename);

double calculateEntropy(const Eigen::SparseMatrix<double> &mat);

double averageRedundancy(const Eigen::SparseMatrix<double> &matrix);

template <typename T, uint8_t compressionLevel>
bool checkMatrixEquality(Eigen::SparseMatrix<T> &eigen);

template <typename T>
void EigenConstructorBenchmark(std::vector<Eigen::Triplet<T>> &eigenTriplet, std::vector<uint64_t> &data, int rows, int cols);

template <typename T>
void CSF1ConstructorBenchmark(std::vector<Eigen::Triplet<T>> &eigenTriplet, std::vector<uint64_t> &data, int rows, int cols);

template <typename T>
void CSF2ConstructorBenchmark(std::vector<Eigen::Triplet<T>> &eigenTriplet, std::vector<uint64_t> &data, int rows, int cols);

template <typename T>
void CSF3ConstructorBenchmark(std::vector<Eigen::Triplet<T>> &eigenTriplet, std::vector<uint64_t> &data, int rows, int cols);

template <typename T>
void ArmadilloConstructorBenchmark(std::vector<Eigen::Triplet<T>> &eigenTriplet, std::vector<uint64_t> &data, int rows, int cols);

template <typename T>
void EigenInnerIteratorBenchmark(Eigen::SparseMatrix<T> eigen, std::vector<uint64_t> &data);

template <typename T>
void CSF1InnerIteratorBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 1> &csf1, std::vector<uint64_t> &data);

template <typename T>
void CSF2InnerIteratorBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 2> &csf2, std::vector<uint64_t> &data);

template <typename T>
void CSF3InnerIteratorBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 3> csf3, std::vector<uint64_t> &data);

template <typename T>
void ArmadilloInnerIteratorBenchmark(arma::sp_mat &mat, std::vector<uint64_t> &data);

template <typename T>
void EigenScalarMultiplicationBenchmark(Eigen::SparseMatrix<T> eigen, std::vector<uint64_t> &data);

template <typename T>
void CSF1ScalarMultiplicationBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 1> csf1, std::vector<uint64_t> &data);

template <typename T>
void CSF2ScalarMultiplicationBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 2> csf2, std::vector<uint64_t> &data);

template <typename T>
void CSF3scalarMultiplicationBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 3> csf3, std::vector<uint64_t> &data);

template <typename T>
void ArmadilloScalarMultiplicationBenchmark(arma::sp_mat mat, std::vector<uint64_t> &data);

template <typename T>
void EigenVectorMultiplicationBenchmark(Eigen::SparseMatrix<T> &eigen, std::vector<uint64_t> &data);

template <typename T>
void CSF1VectorMultiplicationBenchmark(Eigen::SparseMatrix<T> &eigen, IVSparse::SparseMatrix<T, INDEX_TYPE, 1> &csf1, std::vector<uint64_t> &data);

template <typename T>
void CSF2VectorMultiplicationBenchmark(Eigen::SparseMatrix<T> &eigen, IVSparse::SparseMatrix<T, INDEX_TYPE, 2> &csf2, std::vector<uint64_t> &data);

template <typename T>
void CSF3VectorMultiplicationBenchmark(Eigen::SparseMatrix<T> &eigen, IVSparse::SparseMatrix<T, INDEX_TYPE, 3> &csf3, std::vector<uint64_t> &data);

template <typename T>
void ArmadilloVectorMultiplicationBenchmark(arma::sp_mat &mat, std::vector<uint64_t> &data);

template <typename T>
void EigenMemoryFootprintBenchmark(std::vector<uint64_t> &data, std::vector<Eigen::Triplet<T>> &eigenTriplet, uint32_t inner, uint32_t outer);

template <typename T>
void CSF1MemoryFootprintBenchmark(std::vector<uint64_t> &data, std::vector<Eigen::Triplet<T>> &eigenTriplet, uint32_t inner, uint32_t outer);

template <typename T>
void CSF2MemoryFootprintBenchmark(std::vector<uint64_t> &data, std::vector<Eigen::Triplet<T>> &eigenTriplet, uint32_t inner, uint32_t outer);

template <typename T>
void CSF3MemoryFootprintBenchmark(std::vector<uint64_t> &data, std::vector<Eigen::Triplet<T>> &eigenTriplet, uint32_t inner, uint32_t outer);

template <typename T>
void ArmadilloMemoryFootprintBenchmark(std::vector<uint64_t> &data, std::vector<Eigen::Triplet<T>> &eigenTriplet, uint32_t inner, uint32_t outer);

template <typename T>
void eigenTransposeBenchmark(Eigen::SparseMatrix<T> &eigen, std::vector<uint64_t> &data);

template <typename T>
void CSF1TransposeBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 1> &csf1, std::vector<uint64_t> &data);

template <typename T>
void CSF2TransposeBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 2> &csf2, std::vector<uint64_t> &data);

template <typename T>
void CSF3TransposeBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 3> &csf3, std::vector<uint64_t> &data);

template <typename T>
void ArmadilloTransposeBenchmark(arma::sp_mat &mat, std::vector<uint64_t> &data);

template <typename T>
void eigenMatrixMultiplicationBenchmark(Eigen::SparseMatrix<T> &eigen, std::vector<uint64_t> &data);

template <typename T>
void CSF1MatrixMultiplicationBenchmark(Eigen::SparseMatrix<T> &eigen, IVSparse::SparseMatrix<T, INDEX_TYPE, 1> &csf1, std::vector<uint64_t> &data);

template <typename T>
void CSF2MatrixMultiplicationBenchmark(Eigen::SparseMatrix<T> &eigen, IVSparse::SparseMatrix<T, INDEX_TYPE, 2> &csf2, std::vector<uint64_t> &data);

template <typename T>
void CSF3MatrixMultiplicationBenchmark(Eigen::SparseMatrix<T> &eigen, IVSparse::SparseMatrix<T, INDEX_TYPE, 3> &csf3, std::vector<uint64_t> &data);

template <typename T>
void ArmadilloMatrixMultiplicationBenchmark(arma::sp_mat &mat, std::vector<uint64_t> &data);

template <typename T>
void eigenOuterSumBenchmark(Eigen::SparseMatrix<T> &eigen, std::vector<uint64_t> &data);

template <typename T>
void CSF1OuterSumBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 1> &csf1, std::vector<uint64_t> &data);

template <typename T>
void CSF2OuterSumBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 2> &csf2, std::vector<uint64_t> &data);

template <typename T>
void CSF3OuterSumBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 3> &csf3, std::vector<uint64_t> &data);

template <typename T>
void ArmadilloOuterSumBenchmark(arma::sp_mat &mat, std::vector<uint64_t> &data);

template <typename T>
void eigenSumBenchmark(Eigen::SparseMatrix<T> &eigen, std::vector<uint64_t> &data);

template <typename T>
void CSF1SumBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 1> &csf1, std::vector<uint64_t> &data);

template <typename T>
void CSF2SumBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 2> &csf2, std::vector<uint64_t> &data);

template <typename T>
void CSF3SumBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 3> &csf3, std::vector<uint64_t> &data);

template <typename T>
void ArmadilloSumBenchmark(arma::sp_mat &mat, std::vector<uint64_t> &data);

template <typename T>
void eigenNormBenchmark(Eigen::SparseMatrix<T> &eigen, std::vector<uint64_t> &data);

template <typename T>
void CSF1NormBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 1> &csf1, std::vector<uint64_t> &data);

template <typename T>
void CSF2NormBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 2> &csf2, std::vector<uint64_t> &data);

template <typename T>
void CSF3NormBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 3> &csf3, std::vector<uint64_t> &data);

template <typename T>
void ArmadilloNormBenchmark(arma::sp_mat &mat, std::vector<uint64_t> &data);

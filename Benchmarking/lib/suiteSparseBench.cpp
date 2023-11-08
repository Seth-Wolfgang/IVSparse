/**
 * @file benchmark.cpp
 * @author Seth Wolfgang
 * @brief Automated benchmarking for IVSparse and Eigen using matrix market files. 
 *        The main use for this was in benchmarking SuiteSparse matrices and should be 
 *        run with runBenchmark.sh in /Benchmarking.
 * @version 1.0
 * @date 2023-03-19
 *
 * @copyright Copyright (c) 2023
 *
 * Matrix market exmaple code used
 * Source: https://math.nist.gov/MatrixMarket/mmio/c/example_read.c
 * 
 * https://github.com/ginkgo-project/ssget is required for runBenchmark.sh. 
 * The version of ssget provided is slightly modified to store matrices in 
 * /Benchmarking/matrices.
 * 
 * https://arma.sourceforge.net/ is also a reuqirement, but can be commented out
 */

#define NUM_ITERATIONS 5
#include "suiteSparseBenchmarkFunctions.h"

int main(int argc, char** argv) {

    // Checks to make sure the correct number of arguments are passed
    // if (argc < 1) {
    //     fprintf(stderr, "Usage: %s currentFile\n ID", argv[0]);
    //     exit(1);
    // }
    if (argc < 2) {
        argv[1] = (char*)malloc(sizeof(char) * 20);
        strcpy(argv[1], "../testMatrix231.mtx");
        argv[2] = (char*)malloc(sizeof(char));
        argv[2] = "1";
    }

    // Data vectors
    std::vector<Eigen::Triplet<VALUE_TYPE>> eigenTriplet;
    std::vector<uint64_t> data(NUM_OF_BENCHMARKS, 0);
    std::vector<double> matrixData;

    // Records the matrix ID or name

    // if(strstr(argv[2], "mtx") != NULL) {
    // matrixData.push_back(-1);
    // } else {
    matrixData.push_back(std::stoi(argv[2]));
    // std::cout << "Here" << std::endl << std::endl;
    // }

    // Read in the matrix market file
    readFile<VALUE_TYPE>(eigenTriplet, matrixData, argv[1]);
    std::cout << "Matrix Data\tID:" << matrixData[0] << ",  Rows: " << matrixData[1] << ", Cols: " << matrixData[2] << ", Nonzeros: " << matrixData[3] << std::endl;

    Eigen::SparseMatrix<VALUE_TYPE> eigen(matrixData[1], matrixData[2]);
    eigen.reserve(matrixData[3]);
    eigen.setZero();
    eigen.setFromTriplets(eigenTriplet.begin(), eigenTriplet.end());
    eigen.makeCompressed();

    // Create the IVSparse matrices
    IVSparse::SparseMatrix<VALUE_TYPE, INDEX_TYPE, 1> csf1(eigen);
    IVSparse::SparseMatrix<VALUE_TYPE, INDEX_TYPE, 2> csf2(eigen);
    IVSparse::SparseMatrix<VALUE_TYPE, INDEX_TYPE, 3> csf3(eigen);

    // Create the Armadillo matrix
    arma::mat* aMat = new arma::mat(matrixData[1], matrixData[2]);
    for (auto& triplet : eigenTriplet) {
        aMat->at(triplet.row(), triplet.col()) = triplet.value();
    }
    arma::sp_mat armaMat(*aMat);
    delete aMat;

    // Calculate matrix entropy
    matrixData.at(4) = averageRedundancy(eigen);
    BenchAnalysis bench = BenchAnalysis(matrixData);

    // std::cout << eigen << std::endl << std::endl;
    // std::cout << "CSF2 " << csf2 << std::endl << std::endl;
    // std::cout << "IVCSC " << csf3 << std::endl << std::endl;
    std::cout << "Checking CSF 1 Matrix for Equality" << std::endl;
    if (!checkMatrixEquality<VALUE_TYPE, 1>(eigen)) {
        std::cout << "\u001b[4m\u001b[44m Matrix equality failed!\u001b[0m" << std::endl;
        exit(1);
    }

    std::cout << "Checking CSF 2 Matrix for Equality" << std::endl;
    if (!checkMatrixEquality<VALUE_TYPE, 2>(eigen)) {
        std::cout << "\u001b[4m\u001b[44m Matrix equality failed!\u001b[0m" << std::endl;
        exit(1);
    }

    std::cout << "Checking CSF 3 Matrix for Equality" << std::endl;
    if (!checkMatrixEquality<VALUE_TYPE, 3>(eigen)) {
        std::cout << "\u001b[4m\u001b[44m Matrix equality failed!\u001b[0m" << std::endl;
        exit(1);
    }

    // make the line below appear green
    std::cout << "\u001b[32;1;4mMatrix equality passed!\u001b[0m" << std::endl;

    // Random array to select a random benchmark
    int myList[NUM_OF_BENCHMARKS];
    for (int i = 0; i < NUM_OF_BENCHMARKS; i++)
        myList[i] = i;

    int tempList[NUM_OF_BENCHMARKS];

    int currentlySelected = -1;
    int randNum = rand() % NUM_OF_BENCHMARKS;
    int listSize = NUM_OF_BENCHMARKS;

    // Runs each benchmark 100 times
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        mempcpy(&tempList, myList, sizeof(int) * NUM_OF_BENCHMARKS);
        listSize = NUM_OF_BENCHMARKS;

        while (listSize > 0) {
            currentlySelected = -1;

            // Helps to select numbers in a random order
            while (currentlySelected == -1) {
                randNum = rand() % listSize;
                currentlySelected = tempList[randNum];
                if (currentlySelected != -1) {
                    for (int k = randNum; k < listSize - 1; k++)
                        tempList[k] = tempList[k + 1];
                    listSize--;
                }
            }

            // Runs the selected benchmark
            // std::cout << "Running benchmark " << currentlySelected << std::endl;
            switch (currentlySelected) {
            case 0:
                EigenConstructorBenchmark<VALUE_TYPE>(eigenTriplet, data, matrixData[1], matrixData[2]);
                continue;
            case 1:
                CSF1ConstructorBenchmark<VALUE_TYPE>(eigenTriplet, data, matrixData[1], matrixData[2]);
                continue;
            case 2:
                CSF2ConstructorBenchmark<VALUE_TYPE>(eigenTriplet, data, matrixData[1], matrixData[2]);
                continue;
            case 3:
                CSF3ConstructorBenchmark<VALUE_TYPE>(eigenTriplet, data, matrixData[1], matrixData[2]);
                continue;
            case 4:
                ArmadilloConstructorBenchmark<VALUE_TYPE>(eigenTriplet, data, matrixData[1], matrixData[2]);
                continue;
            case 5:
                EigenInnerIteratorBenchmark<VALUE_TYPE>(eigen, data);
                continue;
            case 6:
                CSF1InnerIteratorBenchmark<VALUE_TYPE>(csf1, data);
                continue;
            case 7:
                CSF2InnerIteratorBenchmark<VALUE_TYPE>(csf2, data);
                continue;
            case 8:
                CSF3InnerIteratorBenchmark<VALUE_TYPE>(csf3, data);
                continue;
            case 9:
                ArmadilloInnerIteratorBenchmark<VALUE_TYPE>(armaMat, data);
                continue;
            case 10:
                EigenScalarMultiplicationBenchmark<VALUE_TYPE>(eigen, data);
                continue;
            case 11:
                CSF1ScalarMultiplicationBenchmark<VALUE_TYPE>(csf1, data);
                continue;
            case 12:
                CSF2ScalarMultiplicationBenchmark<VALUE_TYPE>(csf2, data);
                continue;
            case 13:
                CSF3scalarMultiplicationBenchmark<VALUE_TYPE>(csf3, data);
                continue;
            case 14:
                ArmadilloScalarMultiplicationBenchmark<VALUE_TYPE>(armaMat, data);
                continue;
            case 15:
                EigenVectorMultiplicationBenchmark<VALUE_TYPE>(eigen, data);
                continue;
            case 16:
                CSF1VectorMultiplicationBenchmark<VALUE_TYPE>(eigen, csf1, data);
                continue;
            case 17:
                CSF2VectorMultiplicationBenchmark<VALUE_TYPE>(eigen, csf2, data);
                continue;
            case 18:
                CSF3VectorMultiplicationBenchmark<VALUE_TYPE>(eigen, csf3, data);
                continue;
            case 19:
                ArmadilloVectorMultiplicationBenchmark<VALUE_TYPE>(armaMat, data);
                continue;
            case 20:
                EigenMemoryFootprintBenchmark<VALUE_TYPE>(data, eigenTriplet, matrixData[1], matrixData[2]);
                continue;
            case 21:
                CSF1MemoryFootprintBenchmark<VALUE_TYPE>(data, eigenTriplet, matrixData[1], matrixData[2]);
                continue;
            case 22:
                CSF2MemoryFootprintBenchmark<VALUE_TYPE>(data, eigenTriplet, matrixData[1], matrixData[2]);
                continue;
            case 23:
                CSF3MemoryFootprintBenchmark<VALUE_TYPE>(data, eigenTriplet, matrixData[1], matrixData[2]);
                continue;
            case 24:
                ArmadilloMemoryFootprintBenchmark<VALUE_TYPE>(data, eigenTriplet, matrixData[1], matrixData[2]);
                continue;
            case 25:
                eigenTransposeBenchmark<VALUE_TYPE>(eigen, data);
                continue;
            case 26:
                CSF1TransposeBenchmark<VALUE_TYPE>(csf1, data);
                continue;
            case 27:
                CSF2TransposeBenchmark<VALUE_TYPE>(csf2, data);
                continue;
            case 28:
                CSF3TransposeBenchmark<VALUE_TYPE>(csf3, data);
                continue;
            case 29:
                ArmadilloTransposeBenchmark<VALUE_TYPE>(armaMat, data);
                continue;
            case 30:
                eigenMatrixMultiplicationBenchmark<VALUE_TYPE>(eigen, data);
                continue;
            case 31:
                CSF1MatrixMultiplicationBenchmark<VALUE_TYPE>(eigen, csf1, data);
                continue;
            case 32:
                CSF2MatrixMultiplicationBenchmark<VALUE_TYPE>(eigen, csf2, data);
                continue;
            case 33:
                CSF3MatrixMultiplicationBenchmark<VALUE_TYPE>(eigen, csf3, data);
                continue;
            case 34:
                ArmadilloMatrixMultiplicationBenchmark<VALUE_TYPE>(armaMat, data);
                continue;
            case 35:
                eigenOuterSumBenchmark<VALUE_TYPE>(eigen, data);
                continue;
            case 36:
                CSF1OuterSumBenchmark<VALUE_TYPE>(csf1, data);
                continue;
            case 37:
                CSF2OuterSumBenchmark<VALUE_TYPE>(csf2, data);
                continue;
            case 38:
                CSF3OuterSumBenchmark<VALUE_TYPE>(csf3, data);
                continue;
            case 39:
                ArmadilloOuterSumBenchmark<VALUE_TYPE>(armaMat, data);
                continue;
            case 40:
                eigenSumBenchmark<VALUE_TYPE>(eigen, data);
                continue;
            case 41:
                CSF1SumBenchmark<VALUE_TYPE>(csf1, data);
                continue;
            case 42:
                CSF2SumBenchmark<VALUE_TYPE>(csf2, data);
                continue;
            case 43:
                CSF3SumBenchmark<VALUE_TYPE>(csf3, data);
                continue;
            case 44:
                ArmadilloSumBenchmark<VALUE_TYPE>(armaMat, data);
                continue;
            case 45:
                eigenNormBenchmark<VALUE_TYPE>(eigen, data);
                continue;
            case 46:
                CSF1NormBenchmark<VALUE_TYPE>(csf1, data);
                continue;
            case 47:
                CSF2NormBenchmark<VALUE_TYPE>(csf2, data);
                continue;
            case 48:
                CSF3NormBenchmark<VALUE_TYPE>(csf3, data);
                continue;
            case 49:
                ArmadilloNormBenchmark<VALUE_TYPE>(armaMat, data);
                continue;
            }
        }

        bench.pushData(data);
        data.clear();
        data.resize(NUM_OF_BENCHMARKS, 0);
        std::cout << "Iteration " << i << " complete!" << std::endl;
    }

    // Class to calculate the maxes and averages of the benchmarking data
    bench.printTimesToCSV();

    // free(argv[1]);
    // free(argv[2]);
    return 1;
}

/**
 * @brief A procedure to read in a matrix market file and map it to a vector of triplets
 *        This also maps the matrix attributes to a vector
 * @param eigenTriplet
 * @param matrixData
 */
template <typename T>
void readFile(std::vector<Eigen::Triplet<T>>& eigenTriplet, std::vector<double>& matrixData, char* filename) {
    int retCode;
    MM_typecode matcode;
    FILE* f;
    int rows, cols, nonzeros;
    int i, * I, * J;
    double* val;

    // Check for correct number of arguments

    if ((f = fopen(filename, "r")) == NULL) {
        std::cout << "\033[31;1;4mError: Could not open matrix file!\033[0m" << std::endl;
        exit(1);
    }

    // Makes sure the banner can be read
    if (mm_read_banner(f, &matcode) != 0) {
        std::cout << "\033[31;1;4mError: Could not process Matrix Market banner.\033[0m" << std::endl;
        exit(1);
    }

    // Makes sure the matrix is not complex
    if (mm_is_complex(matcode)) { //|| mm_is_pattern(matcode)) {
        std::cout << "\033[31;1;4mError: This application does not support \033[0m" << std::endl;
        std::cout << "\033[31;1;4mMarket Market type: \033[0m" << mm_typecode_to_str(matcode) << std::endl;
        std::cout << "Matrix might be complex or not a matrix";
        exit(1);
    }

    // Reads the dimensions and number of nonzeros
    if ((retCode = mm_read_mtx_crd_size(f, &rows, &cols, &nonzeros)) != 0) {
        std::cout << "\033[31;1;4mError: Could not read matrix dimensions.\033[0m" << std::endl;
        exit(1);
    }

    if (nonzeros > 400000) {
        std::cout << "\033[31;1;4mMatrix too large, skipping...\033[0m" << std::endl;
        exit(1);
    }

    // Allocate memory for the matrix
    I = (int*)malloc(nonzeros * sizeof(int));
    J = (int*)malloc(nonzeros * sizeof(int));
    val = (double*)calloc(nonzeros, sizeof(double));
    // Read the matrix
    if (mm_is_pattern(matcode)) {
        for (i = 0; i < nonzeros; i++) {
            fscanf(f, "%d %d\n", &I[i], &J[i]);
            val[i] = 1.0;
            I[i]--; /* adjust from 1-based to 0-based */
            J[i]--;
        }
    }
    else {
        for (i = 0; i < nonzeros; i++) {
            fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
            I[i]--; /* adjust from 1-based to 0-based */
            J[i]--;
        }
    }

    // Close the file
    if (f != stdin)
        fclose(f);

    // Create the Eigen triplet
    for (int i = 0; i < nonzeros; i++) {
        eigenTriplet.push_back(Eigen::Triplet<VALUE_TYPE>(I[i], J[i], val[i]));
    }

    // Create the matrix data
    matrixData.push_back(rows);
    matrixData.push_back(cols);
    matrixData.push_back(nonzeros);

    // use a hash to store unique values

    // iterates through the values and inserts them into the hash to record unique values
    //  std::unordered_set<double> uniqueValues;
    //  for (int i = 0; i < nonzeros; i++) {
    //      uniqueValues.insert(val[i]);
    //  }
    //  matrixData.push_back(1.0 - ((double)uniqueValues.size() / nonzeros));

    matrixData.push_back(0);

    // Calculate matrix density
    matrixData.push_back((double)nonzeros / (double)(rows * cols));

    // Free the memory
    free(I);
    free(J);
    free(val);
}

// https://www.phind.com/search?cache=e2eb22ba-3690-4155-a8f1-18a61dad9cbe
//  "Calculate entropy of an Eigen::Sparse matrix in C++"
//  Plus some modifications
double calculateEntropy(const Eigen::SparseMatrix<double>& matrix) {
    std::vector<double> values;
    values.reserve(matrix.nonZeros());

    // Extract values from the sparse matrix
    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
            values.push_back(it.value());
        }
    }

    // Count occurrences of each value
    std::unordered_map<double, unsigned> counts;
    for (double value : values) {
        ++counts[value];
    }

    // Calculate probabilities of each value
    const double num_samples = static_cast<double>(values.size());
    std::vector<double> probabilities;
    probabilities.reserve(counts.size());
    for (const auto& pair : counts) {
        double probability = static_cast<double>(pair.second) / num_samples;
        probabilities.push_back(probability);
    }

    // Calculate Shannon entropy
    double entropy = 0.0;
    for (double probability : probabilities) {
        entropy -= probability * std::log2(probability);
    }

    return entropy;
}

// calculates the redandncy per column
double averageRedundancy(const Eigen::SparseMatrix<double>& matrix) {
    const int numRows = matrix.rows();
    const int numCols = matrix.cols();
    double totalRedundancy = 0.0;
    std::vector<double> uniqueValues;

    for (int j = 0; j < numCols; ++j) {
        Eigen::SparseVector<double> column = matrix.col(j);
        if (column.nonZeros() == 0) {
            continue;
        }

        std::unordered_set<double> uniqueValues;
        for (typename Eigen::SparseVector<double>::InnerIterator it(column); it; ++it) {
            uniqueValues.insert(it.value());
        }

        double totalValues = static_cast<double>(column.nonZeros());
        double redundancy = 1.0 - (uniqueValues.size() / totalValues);
        totalRedundancy += redundancy;
    }
    std::cout << "TR: " << totalRedundancy << " NC: " << numCols << std::endl;

    return totalRedundancy / static_cast<double>(numCols);
}

template <typename T, uint8_t compressionLevel>
bool checkMatrixEquality(Eigen::SparseMatrix<T>& eigen) {
    IVSparse::SparseMatrix<T, INDEX_TYPE, compressionLevel> csf(eigen);

    // Checking basic attributes
    if (eigen.outerSize() != csf.outerSize() || eigen.outerSize() != csf.outerSize() || eigen.innerSize() != csf.innerSize() || eigen.nonZeros() != csf.nonZeros()) {
        std::cout << "\u001b[31;1;4mError: Matrix Comparison Returned False!\u001b[0m" << std::endl;
        std::cout << "Outer size mismatch!" << std::endl;
        return false;
    }

    // Now we start creating the full representation of each matrix
    T** eigenMatrix = new T * [eigen.rows()];
    T** csfMatrix = new T * [csf.rows()];

    // Initialize the matrices
    for (size_t i = 0; i < eigen.rows(); i++) {
        eigenMatrix[i] = new T[eigen.cols()];
        memset(eigenMatrix[i], 0, sizeof(eigenMatrix[i]) * eigen.cols());

        csfMatrix[i] = new T[csf.cols()];
        memset(csfMatrix[i], 0, sizeof(csfMatrix[i]) * csf.cols());
    }

    // Build the full matrix representation of each matrix
    for (size_t i = 0; i < eigen.cols(); i++) {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(eigen, i); it; ++it) {
            eigenMatrix[it.row()][it.col()] = it.value();
        }

        for (typename IVSparse::SparseMatrix<T, INDEX_TYPE, compressionLevel>::InnerIterator it(csf, i); it; ++it) {
            csfMatrix[it.row()][it.col()] = it.value();
        }
    }

    // Finally compare the uncompressed matrices
    for (size_t i = 0; i < eigen.rows(); i++) {
        for (size_t j = 0; j < eigen.cols(); j++) {
            if (eigenMatrix[i][j] != csfMatrix[i][j]) {
                std::cout << "\u001b[31;1;4mError: Matrix Comparison Returned False!\u001b[0m" << std::endl;
                std::cout << "At: (" << i << ", " << j << ")" << std::endl;
                std::cout << "Eigen: " << eigenMatrix[i][j] << " csf: " << csfMatrix[i][j] << std::endl;
                return false;
            }
        }
    }

    // Compare the compressed matrices with the uncompressed
    for (size_t i = 0; i < eigen.cols(); i++) {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(eigen, i); it; ++it) {
            if (csfMatrix[it.row()][it.col()] != it.value() || eigenMatrix[it.row()][it.col()] != it.value()) {
                std::cout << "\u001b[31;1;4mError: Matrix Comparison Returned False!\u001b[0m" << std::endl;
                std::cout << "At: (" << it.row() << ", " << it.col() << ")" << std::endl;
                std::cout << "Eigen: " << eigenMatrix[it.row()][it.col()] << " csf: " << it.value() << std::endl;
                return false;
            }
        }

        for (typename IVSparse::SparseMatrix<T, INDEX_TYPE, compressionLevel>::InnerIterator it(csf, i); it; ++it) {
            if (csfMatrix[it.row()][it.col()] != it.value() || eigenMatrix[it.row()][it.col()] != it.value()) {
                std::cout << "\u001b[31;1;4mError: Matrix Comparison Returned False!\u001b[0m" << std::endl;
                std::cout << "At: (" << it.row() << ", " << it.col() << ")" << std::endl;
                std::cout << "Eigen: " << eigenMatrix[it.row()][it.col()] << " CSF: " << it.value() << std::endl;
                return false;
            }
        }
    }

    // Free the memory
    for (size_t i = 0; i < eigen.rows(); i++) {
        delete[] csfMatrix[i];
        delete[] eigenMatrix[i];
    }
    delete[] csfMatrix;
    delete[] eigenMatrix;

    return true;
}

/**
 * @brief Benchmark for the Eigen::SparseMatrix constructor
 *
 */

template <typename T>
void EigenConstructorBenchmark(std::vector<Eigen::Triplet<T>>& eigenTriplet, std::vector<uint64_t>& data, int rows, int cols) {

    // Timer
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // benchmark the Eigen constructor
    start = std::chrono::system_clock::now();
    Eigen::SparseMatrix<T> eigenMat(rows, cols);
    eigenMat.setFromTriplets(eigenTriplet.begin(), eigenTriplet.end());
    eigenMat.makeCompressed();
    end = std::chrono::system_clock::now();
    data.at(0) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the CSC constructor
 */

template <typename T>
void CSF1ConstructorBenchmark(std::vector<Eigen::Triplet<T>>& eigenTriplet, std::vector<uint64_t>& data, int rows, int cols) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::vector<std::tuple<int, int, T>> csf1Data;

    for (int i = 0; i < eigenTriplet.size(); i++) {
        csf1Data.push_back(std::make_tuple(eigenTriplet[i].row(), eigenTriplet[i].col(), eigenTriplet[i].value()));
    }

    // benchmark the CSC constructor
    start = std::chrono::system_clock::now();
    IVSparse::SparseMatrix<T, int, 1> csf1(csf1Data, rows, cols, eigenTriplet.size());
    end = std::chrono::system_clock::now();

    data.at(1) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the CSF2 constructor
 */

template <typename T>
void CSF2ConstructorBenchmark(std::vector<Eigen::Triplet<T>>& eigenTriplet, std::vector<uint64_t>& data, int rows, int cols) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::vector<std::tuple<int, int, T>> csf2Data;

    for (int i = 0; i < eigenTriplet.size(); i++) {
        csf2Data.push_back(std::make_tuple(eigenTriplet[i].row(), eigenTriplet[i].col(), eigenTriplet[i].value()));
    }

    // benchmark the CSF2 constructor
    start = std::chrono::system_clock::now();
    IVSparse::SparseMatrix<T, int, 2> csf2(csf2Data, rows, cols, eigenTriplet.size());
    end = std::chrono::system_clock::now();

    data.at(2) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the IVCSC constructor
 */

template <typename T>
void CSF3ConstructorBenchmark(std::vector<Eigen::Triplet<T>>& eigenTriplet, std::vector<uint64_t>& data, int rows, int cols) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::vector<std::tuple<int, int, T>> csf3Data;

    for (int i = 0; i < eigenTriplet.size(); i++) {
        csf3Data.push_back(std::make_tuple(eigenTriplet[i].row(), eigenTriplet[i].col(), eigenTriplet[i].value()));
    }

    // benchmark the CSF2 constructor
    start = std::chrono::system_clock::now();
    IVSparse::SparseMatrix<T, int, 3> csf3(csf3Data, rows, cols, eigenTriplet.size());
    end = std::chrono::system_clock::now();

    data.at(3) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the Armadillo constructor
 */

template <typename T>
void ArmadilloConstructorBenchmark(std::vector<Eigen::Triplet<T>>& eigenTriplet, std::vector<uint64_t>& data, int rows, int cols) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // benchmark the Eigen constructor
    start = std::chrono::system_clock::now();
    arma::sp_mat armaMat(rows, cols);
    for (auto& triplet : eigenTriplet) {
        armaMat(triplet.row(), triplet.col()) = triplet.value();
    }
    end = std::chrono::system_clock::now();
    data.at(4) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the Eigen::SparseMatrix InnerIterator
 *
 */

template <typename T>
void EigenInnerIteratorBenchmark(Eigen::SparseMatrix<T> eigen, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    VALUE_TYPE total = 0;

    // Eigen
    start = std::chrono::system_clock::now();
    for (uint32_t i = 0; i < eigen.outerSize(); ++i) {
        for (Eigen::SparseMatrix<VALUE_TYPE>::InnerIterator it(eigen, i); it; ++it) {
            total += it.value();
            // std::cout << it.value() << ",";
        }
    }
    end = std::chrono::system_clock::now();
    std::cout << "\nEigen: " << total << std::endl;

    data.at(5) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the CSC InnerIterator
 */

template <typename T>
void CSF1InnerIteratorBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 1>& csf1, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    VALUE_TYPE total = 0;

    // CSF 1
    start = std::chrono::system_clock::now();
    for (uint32_t i = 0; i < csf1.outerSize(); ++i) {
        for (typename IVSparse::SparseMatrix<VALUE_TYPE, INDEX_TYPE, 1>::InnerIterator it(csf1, i); it; ++it) {
            total += it.value();
        }
    }
    end = std::chrono::system_clock::now();
    std::cout << "\nCSF 1: " << total << std::endl;

    data.at(6) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the CSF2 InnerIterator
 *
 */

template <typename T>
void CSF2InnerIteratorBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 2>& csf2, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    VALUE_TYPE total = 0;

    // CSF 2
    start = std::chrono::system_clock::now();
    for (uint32_t i = 0; i < csf2.outerSize(); ++i) {
        for (typename IVSparse::SparseMatrix<VALUE_TYPE, INDEX_TYPE, 2>::InnerIterator it(csf2, i); it; ++it) {
            total += it.value();
        }
    }
    end = std::chrono::system_clock::now();
    std::cout << "\nCSF 2: " << total << std::endl;

    data.at(7) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the IVCSC InnerIterator
 *
 */

template <typename T>
void CSF3InnerIteratorBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 3> csf3, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    VALUE_TYPE total = 0;

    // CSF 3
    start = std::chrono::system_clock::now();
    for (uint32_t i = 0; i < csf3.outerSize(); ++i) {
        for (typename IVSparse::SparseMatrix<double, INDEX_TYPE, 3>::InnerIterator it(csf3, i); it; ++it) {
            total += it.value();
        }
    }
    end = std::chrono::system_clock::now();
    std::cout << "\nCSF 3: " << total << std::endl;
    data.at(8) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the Armadillo InnerIterator
 */

template <typename T>
void ArmadilloInnerIteratorBenchmark(arma::sp_mat& armaMat, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    VALUE_TYPE total = 0;

    start = std::chrono::system_clock::now();
    for (int i = 0; i < armaMat.n_cols; ++i) {
        for (arma::sp_mat::iterator it = armaMat.begin_col(i); it != armaMat.end_col(i); ++it) {
            total += *it;
        }
    }
    end = std::chrono::system_clock::now();
    std::cout << "\nArmadillo: " << total << std::endl;
    data.at(9) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the Eigen::SparseMatrix Scalar Multiplication
 *
 */

template <typename T>
void EigenScalarMultiplicationBenchmark(Eigen::SparseMatrix<T> eigen, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // Eigen
    start = std::chrono::system_clock::now();
    eigen *= 2;
    end = std::chrono::system_clock::now();

    data.at(10) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the CSC Scalar Multiplication
 */

template <typename T>
void CSF1ScalarMultiplicationBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 1> csf1, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // CSF 2
    start = std::chrono::system_clock::now();
    csf1 *= 2;
    end = std::chrono::system_clock::now();
    data.at(11) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the CSF2 Scalar Multiplication
 *
 */

template <typename T>
void CSF2ScalarMultiplicationBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 2> csf2, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // CSF 2
    start = std::chrono::system_clock::now();
    csf2 *= 2;
    end = std::chrono::system_clock::now();
    data.at(12) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the IVCSC Scalar Multiplication
 *
 */

template <typename T>
void CSF3scalarMultiplicationBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 3> csf3, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // CSF 3
    start = std::chrono::system_clock::now();
    csf3 *= 2;
    // csf3 = csf3 * 2;
    end = std::chrono::system_clock::now();

    data.at(13) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the Armadillo Scalar Multiplication
 */

template <typename T>
void ArmadilloScalarMultiplicationBenchmark(arma::sp_mat armaMat, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // Eigen
    start = std::chrono::system_clock::now();
    armaMat *= 2;
    end = std::chrono::system_clock::now();

    data.at(14) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the Eigen::SparseMatrix Vector Multiplication
 *
 */

template <typename T>
void EigenVectorMultiplicationBenchmark(Eigen::SparseMatrix<T>& eigen, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // Eigen Vectors
    Eigen::SparseMatrix<T> eigenTempVector(eigen.cols(), 1);
    Eigen::VectorXd eigenVector(eigen.cols());
    eigenTempVector.reserve(eigen.cols());

    // Filling eigen vectors
    for (int i = 0; i < eigen.cols(); ++i) {
        eigenTempVector.insert(i, 0) = 2;
        eigenVector(i) = 2.0;
    }

    // Eigen
    start = std::chrono::system_clock::now();
    Eigen::MatrixXd resultEigen = eigen * eigenVector;
    end = std::chrono::system_clock::now();
    data.at(15) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the CSC Vector Multiplication
 */

template <typename T>
void CSF1VectorMultiplicationBenchmark(Eigen::SparseMatrix<T>& eigen, IVSparse::SparseMatrix<T, INDEX_TYPE, 1>& csf1, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // Eigen Vectors
    Eigen::VectorXd eigenTempVector(eigen.cols());
    Eigen::VectorXd result;

    // Filling eigen vectors
    eigenTempVector.fill(2);

    // CSF 2
    start = std::chrono::system_clock::now();
    result = csf1 * eigenTempVector;
    end = std::chrono::system_clock::now();

    data.at(16) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the CSF2 Vector Multiplication
 *
 */

template <typename T>
void CSF2VectorMultiplicationBenchmark(Eigen::SparseMatrix<T>& eigen, IVSparse::SparseMatrix<T, INDEX_TYPE, 2>& csf2, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // Eigen Vectors
    Eigen::VectorXd eigenTempVector(eigen.cols());
    Eigen::VectorXd result;

    // Filling eigen vectors
    eigenTempVector.fill(2);

    // CSF 2
    start = std::chrono::system_clock::now();
    result = csf2 * eigenTempVector;
    end = std::chrono::system_clock::now();

    data.at(17) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the IVCSC Vector Multiplication
 *
 */

template <typename T>
void CSF3VectorMultiplicationBenchmark(Eigen::SparseMatrix<T>& eigen, IVSparse::SparseMatrix<T, INDEX_TYPE, 3>& csf3, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // Eigen Vectors
    Eigen::VectorXd eigenTempVector(eigen.cols());
    Eigen::VectorXd result;

    // Filling eigen vectors
    eigenTempVector.fill(2);

    // CSF 2
    start = std::chrono::system_clock::now();
    result = csf3 * eigenTempVector;
    end = std::chrono::system_clock::now();

    data.at(18) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the Armadillo Vector Multiplication
 */

template <typename T>
void ArmadilloVectorMultiplicationBenchmark(arma::sp_mat& armaMat, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // Eigen Vectors
    arma::vec armaTempVector(armaMat.n_cols);
    arma::vec result;

    // Filling eigen vectors
    armaTempVector.fill(2);

    // Eigen
    start = std::chrono::system_clock::now();
    result = armaMat * armaTempVector;
    end = std::chrono::system_clock::now();

    data.at(19) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the Eigen::SparseMatrix Memory Footprint Benchmark
 *
 */

template <typename T>
void EigenMemoryFootprintBenchmark(std::vector<uint64_t>& data, std::vector<Eigen::Triplet<T>>& eigenTriplet, uint32_t inner, uint32_t outer) {
    // Create Eigen Matrix
    Eigen::SparseMatrix<VALUE_TYPE> eigenMatrix(inner, outer);
    eigenMatrix.setFromTriplets(eigenTriplet.begin(), eigenTriplet.end());
    eigenMatrix.makeCompressed();
    data.at(20) = eigenMatrix.nonZeros() * sizeof(double) + eigenMatrix.nonZeros() * sizeof(uint32_t) + (eigenMatrix.outerSize() + 1) * sizeof(uint32_t);
}

/**
 * @brief CSC Memory Footprint Benchmark
 */

template <typename T>
void CSF1MemoryFootprintBenchmark(std::vector<uint64_t>& data, std::vector<Eigen::Triplet<T>>& eigenTriplet, uint32_t inner, uint32_t outer) {
    // Create Eigen Matrix
    Eigen::SparseMatrix<VALUE_TYPE> eigenMatrix(inner, outer);
    eigenMatrix.setFromTriplets(eigenTriplet.begin(), eigenTriplet.end());
    eigenMatrix.makeCompressed();

    // CSF Construction
    IVSparse::SparseMatrix<VALUE_TYPE, INDEX_TYPE, 1> csf1Matrix(eigenMatrix);
    data.at(21) = csf1Matrix.byteSize();
}

/**
 * @brief CSF2 Memory Footprint Benchmark
 *
 */

template <typename T>
void CSF2MemoryFootprintBenchmark(std::vector<uint64_t>& data, std::vector<Eigen::Triplet<T>>& eigenTriplet, uint32_t inner, uint32_t outer) {
    // Create Eigen Matrix
    Eigen::SparseMatrix<VALUE_TYPE> eigenMatrix(inner, outer);
    eigenMatrix.setFromTriplets(eigenTriplet.begin(), eigenTriplet.end());
    eigenMatrix.makeCompressed();

    // CSF Construction
    IVSparse::SparseMatrix<VALUE_TYPE, INDEX_TYPE, 2> csf2Matrix(eigenMatrix);
    data.at(22) = csf2Matrix.byteSize();
}

/**
 * @brief IVCSC Memory Footprint Benchmark
 *
 */

template <typename T>
void CSF3MemoryFootprintBenchmark(std::vector<uint64_t>& data, std::vector<Eigen::Triplet<T>>& eigenTriplet, uint32_t inner, uint32_t outer) {
    // Create Eigen Matrix
    Eigen::SparseMatrix<VALUE_TYPE> eigenMatrix(inner, outer);
    eigenMatrix.setFromTriplets(eigenTriplet.begin(), eigenTriplet.end());
    eigenMatrix.makeCompressed();

    // CSF Construction
    IVSparse::SparseMatrix<VALUE_TYPE, INDEX_TYPE, 3> csf3Matrix(eigenMatrix);
    data.at(23) = csf3Matrix.byteSize();
}

/**
 * @brief Armadillo Memory Footprint Benchmark
 */

template <typename T>
void ArmadilloMemoryFootprintBenchmark(std::vector<uint64_t>& data, std::vector<Eigen::Triplet<T>>& eigenTriplet, uint32_t inner, uint32_t outer) {
    arma::mat* aMat = new arma::mat(inner, outer);
    for (auto& triplet : eigenTriplet) {
        aMat->at(triplet.row(), triplet.col()) = triplet.value();
    }
    arma::sp_mat armaMatrix(*aMat);

    // Same as Eigen, both are CSC
    data.at(24) = armaMatrix.n_nonzero * sizeof(double) + armaMatrix.n_nonzero * sizeof(uint32_t) + (armaMatrix.n_cols + 1) * sizeof(uint32_t);
}

/**
 * @brief Benchmark for transpose method of Eigen::SparseMatrix
 *
 */

template <typename T>
void eigenTransposeBenchmark(Eigen::SparseMatrix<T>& eigen, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // Eigen
    start = std::chrono::system_clock::now();
    Eigen::SparseMatrix<T> resultEigen = eigen.transpose();
    end = std::chrono::system_clock::now();
    data.at(25) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for transpose method of IVSparse::SparseMatrix
 */

template <typename T>
void CSF1TransposeBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 1>& csf1, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // CSF 2
    start = std::chrono::system_clock::now();
    IVSparse::SparseMatrix<T, INDEX_TYPE, 1> resultCSF1 = csf1.transpose();
    end = std::chrono::system_clock::now();

    data.at(26) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for transpose method of IVSparse::SparseMatrix
 *
 */

template <typename T>
void CSF2TransposeBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 2>& csf2, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // CSF 2
    start = std::chrono::system_clock::now();
    IVSparse::SparseMatrix<T, INDEX_TYPE, 2> resultCSF2 = csf2.transpose();
    end = std::chrono::system_clock::now();

    data.at(27) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for transpose method of IVSparse::SparseMatrix
 *
 *
 */

template <typename T>
void CSF3TransposeBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 3>& csf3, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // CSF 3
    start = std::chrono::system_clock::now();
    IVSparse::SparseMatrix<T, INDEX_TYPE, 3> resultCSF3 = csf3.transpose();
    end = std::chrono::system_clock::now();

    data.at(28) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for transpose method of Armadillo
 */

template <typename T>
void ArmadilloTransposeBenchmark(arma::sp_mat& armaMat, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // Eigen
    start = std::chrono::system_clock::now();
    arma::sp_mat result = armaMat.t();
    end = std::chrono::system_clock::now();

    data.at(29) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for matrix multiplication of Eigen::SparseMatrix
 */

template <typename T>
void eigenMatrixMultiplicationBenchmark(Eigen::SparseMatrix<T>& eigen, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    Eigen::Matrix<T, -1, -1> eigenT = Eigen::Matrix<T, -1, -1>::Ones(eigen.cols(), eigen.rows());

    // Eigen
    start = std::chrono::system_clock::now();
    Eigen::Matrix<T, -1, -1> resultEigen = eigen * eigenT;
    end = std::chrono::system_clock::now();

    data.at(30) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for matrix multiplication of IVSparse::SparseMatrix
 */

template <typename T>
void CSF1MatrixMultiplicationBenchmark(Eigen::SparseMatrix<T>& eigen, IVSparse::SparseMatrix<T, INDEX_TYPE, 1>& csf1, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    Eigen::Matrix<T, -1, -1> eigenT = Eigen::Matrix<T, -1, -1>::Ones(eigen.cols(), eigen.rows());

    // CSF 2
    start = std::chrono::system_clock::now();
    Eigen::Matrix<T, -1, -1> result = csf1 * eigenT;
    end = std::chrono::system_clock::now();

    data.at(31) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for matrix multiplication of IVSparse::SparseMatrix
 */

template <typename T>
void CSF2MatrixMultiplicationBenchmark(Eigen::SparseMatrix<T>& eigen, IVSparse::SparseMatrix<T, INDEX_TYPE, 2>& csf2, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    Eigen::Matrix<T, -1, -1> eigenT = Eigen::Matrix<T, -1, -1>::Ones(eigen.cols(), eigen.rows());

    // CSF 2
    start = std::chrono::system_clock::now();
    Eigen::Matrix<T, -1, -1> result = csf2 * eigenT;
    end = std::chrono::system_clock::now();

    data.at(32) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for matrix multiplication of IVSparse::SparseMatrix
 */

template <typename T>
void CSF3MatrixMultiplicationBenchmark(Eigen::SparseMatrix<T>& eigen, IVSparse::SparseMatrix<T, INDEX_TYPE, 3>& csf3, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    Eigen::Matrix<T, -1, -1> eigenT = Eigen::Matrix<T, -1, -1>::Ones(eigen.cols(), eigen.rows());

    // CSF 3
    start = std::chrono::system_clock::now();
    Eigen::Matrix<T, -1, -1> result = csf3 * eigenT;
    end = std::chrono::system_clock::now();

    data.at(33) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for matrix multiplication of Armadillo
 */

template <typename T>
void ArmadilloMatrixMultiplicationBenchmark(arma::sp_mat& armaMat, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    arma::mat ones = arma::ones(armaMat.n_cols, armaMat.n_rows);

    // Eigen
    start = std::chrono::system_clock::now();
    arma::mat result = armaMat * ones;
    end = std::chrono::system_clock::now();

    data.at(34) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the Eigen::SparseMatrix OuterSum
 */

template <typename T>
void eigenOuterSumBenchmark(Eigen::SparseMatrix<T>& eigen, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // Eigen
    start = std::chrono::system_clock::now();
    Eigen::VectorXd resultEigen(eigen.cols());
    for (int i = 0; i < eigen.cols(); ++i) {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(eigen, i); it; ++it) {
            resultEigen(i) += it.value();
        }
    }
    end = std::chrono::system_clock::now();

    data.at(35) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the CSC OuterSum
 */

template <typename T>
void CSF1OuterSumBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 1>& csf1, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // CSF 2
    start = std::chrono::system_clock::now();
    std::vector<T> resultCSF2 = csf1.outerSum();
    end = std::chrono::system_clock::now();

    data.at(36) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the CSF2 OuterSum
 */

template <typename T>
void CSF2OuterSumBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 2>& csf2, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // CSF 2
    start = std::chrono::system_clock::now();
    std::vector<T> resultCSF2 = csf2.outerSum();
    end = std::chrono::system_clock::now();

    data.at(37) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the IVCSC OuterSum
 */

template <typename T>
void CSF3OuterSumBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 3>& csf3, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    // CSF 3
    start = std::chrono::system_clock::now();
    std::vector<T> resultCSF3 = csf3.outerSum();
    end = std::chrono::system_clock::now();

    data.at(38) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the Armadillo OuterSum
 */

template <typename T>
void ArmadilloOuterSumBenchmark(arma::sp_mat& armaMat, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    arma::vec result(armaMat.n_cols);

    // Eigen
    start = std::chrono::system_clock::now();

    for (int i = 0; i < armaMat.n_cols; ++i) {
        result(i) = arma::accu(armaMat.col(i));
    }

    end = std::chrono::system_clock::now();

    data.at(39) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the Eigen::SparseMatrix Sum
 */

template <typename T>
void eigenSumBenchmark(Eigen::SparseMatrix<T>& eigen, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    T resultEigen(eigen.rows());

    // Eigen
    start = std::chrono::system_clock::now();
    resultEigen = eigen.sum();
    end = std::chrono::system_clock::now();

    std::cout << "\nEigen: " << resultEigen << std::endl;
    data.at(40) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the CSC Sum
 */

template <typename T>
void CSF1SumBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 1>& csf1, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    T resultCSF2;

    // CSF 2
    start = std::chrono::system_clock::now();
    resultCSF2 = csf1.sum();
    end = std::chrono::system_clock::now();

    std::cout << "\nCSF 1: " << resultCSF2 << std::endl;
    data.at(41) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the CSF2 Sum
 */

template <typename T>
void CSF2SumBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 2>& csf2, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    T resultCSF2;

    // CSF 2
    start = std::chrono::system_clock::now();
    resultCSF2 = csf2.sum();
    end = std::chrono::system_clock::now();

    std::cout << "\nCSF 2: " << resultCSF2 << std::endl;
    data.at(42) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the IVCSC Sum
 */

template <typename T>
void CSF3SumBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 3>& csf3, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    T resultCSF3;

    // CSF 3
    start = std::chrono::system_clock::now();
    resultCSF3 = csf3.sum();
    end = std::chrono::system_clock::now();

    std::cout << "\nCSF 3: " << resultCSF3 << std::endl;
    data.at(43) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the Armadillo Sum
 */

template <typename T>
void ArmadilloSumBenchmark(arma::sp_mat& armaMat, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    T result;

    start = std::chrono::system_clock::now();
    result = arma::accu(armaMat);
    end = std::chrono::system_clock::now();

    std::cout << "\nArmadillo: " << result << std::endl;
    data.at(44) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the Eigen::SparseMatrix Norm
 */

template <typename T>
void eigenNormBenchmark(Eigen::SparseMatrix<T>& eigen, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    double resultEigen(eigen.rows());

    // Eigen
    start = std::chrono::system_clock::now();
    resultEigen = eigen.norm();
    end = std::chrono::system_clock::now();

    std::cout << "\nEigen: " << resultEigen << std::endl;
    data.at(45) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the CSC Norm
 */

template <typename T>
void CSF1NormBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 1>& csf1, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    double resultCSF2;

    // CSF 2
    start = std::chrono::system_clock::now();
    resultCSF2 = csf1.norm();
    end = std::chrono::system_clock::now();

    std::cout << "\nCSF 1: " << resultCSF2 << std::endl;
    data.at(46) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the CSF2 Norm
 */

template <typename T>
void CSF2NormBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 2>& csf2, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    double resultCSF2;

    // CSF 2
    start = std::chrono::system_clock::now();
    resultCSF2 = csf2.norm();
    end = std::chrono::system_clock::now();

    std::cout << "\nCSF 2: " << resultCSF2 << std::endl;
    data.at(47) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the IVCSC Norm
 */

template <typename T>
void CSF3NormBenchmark(IVSparse::SparseMatrix<T, INDEX_TYPE, 3>& csf3, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    double resultCSF3;

    // CSF 3
    start = std::chrono::system_clock::now();
    resultCSF3 = csf3.norm();
    end = std::chrono::system_clock::now();

    std::cout << "\nCSF 3: " << resultCSF3 << std::endl;
    data.at(48) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

/**
 * @brief Benchmark for the Armadillo Norm
 */

template <typename T>
void ArmadilloNormBenchmark(arma::sp_mat& armaMat, std::vector<uint64_t>& data) {
    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = std::chrono::system_clock::now();
    double result = arma::norm(armaMat, "fro");
    end = std::chrono::system_clock::now();

    std::cout << "\nArmadillo: " << result << std::endl;
    data.at(49) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}
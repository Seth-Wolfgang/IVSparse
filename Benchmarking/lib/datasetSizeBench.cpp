/**
 * @file datasetSizeBench.cpp
 * @author Seth Wolfgang
 * @brief This file is used to benchmark the size of the CSC and IVSparse matrices. This file
 *        was specifically made to work with real datasets of matrix market format or text files
 *        containing only coordinates and values. There is a generate matrix method provided, but 
 *        its call in benchmark() is commented out.
 * 
 *        COO csv files should look like:
 *         1,1,1
 *         3,4,3
 *         4,6,4
 *         only the coordinates should be listed
 *         with no metadata.
 * 
 * @date 2023-08-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */



#include "benchmarkFunctions.h"
#include "../../misc/matrix_creator.cpp"
#include <tuple>
#include <unordered_map>
#include <functional>
#include <string>
#include <fstream>
#include <iostream>

void benchmark(char* filepath, std::function<double(std::string, std::unordered_map<std::string, double>&)> func);
void loadMatrix(std::vector<std::tuple<uint, uint, double>>& data, std::function<double(std::string, std::unordered_map<std::string, double>&)> func, char* filepath);
double returnDouble(std::string val, std::unordered_map<std::string, double>& myMap);
double classifyDouble(std::string val, std::unordered_map<std::string, double>& myMap);
void load_mm_matrix(std::vector<std::tuple<uint, uint, double>>& data, char* filename);
void generateMatrix(std::vector<std::tuple<uint, uint, double>>& data, int numRows, int numCols, int sparsity, uint64_t seed, uint64_t maxValue);
template <typename T, typename indexType, int compressionLevel>
uint64_t buildMatrix(std::vector<std::tuple<uint, uint, double>>& data, uint rows, uint cols, uint nnz);

template <typename T, typename indexType, int compressionLevel>
void averageRedundancy(IVSparse::SparseMatrix<T, indexType, compressionLevel>& matrix);

#define ITERATIONS 10

int main() {

    char* path = "../datasets/PR02R.mtx"; // comes from 10X genmatics, see bibliography from the paper for this librar for more info.

     //returnDouble is a first class function. Some data sets use strings (like survey data: "very likely, likely..."), 
     //so we can classify them by using the other first class function classifyDouble.
    benchmark(path, returnDouble);
    // benchmark("../datasets/tags.csv", classifyDouble);  // tags.csv comes from movieLens 25ml dataset

    return 0;
}

// This reads in a COO matrix and creates a vector of tuples to store it.
// The tuples are in the form of (row, col, value) and values are seperated by commas (.csv).
// The first line of the input file is the first tuple of the matrix. The second line is the second tuple, etc.
void loadMatrix(std::vector<std::tuple<uint, uint, double>>& data, std::function<double(std::string, std::unordered_map<std::string, double>&)> func, char* filename) {
    std::unordered_map<std::string, double> map;

    FILE* file;
    file = fopen(filename, "r");
    char line[1024];

    uint i = 0;
    while (fgets(line, 1024, file)) {
        // split the row into 3 seperate values
        char* val1 = strtok(line, ",");
        char* val2 = strtok(NULL, ",");
        char* val3 = strtok(NULL, ",");
        std::string val3String(val3);

        // add the values to the data vector
        data.push_back(std::make_tuple(atoi(val2), atoi(val1), func(val3String, map))); //val1 and val2 may need to be switched depending on data set.
    }

    // copy data to data vector
    fclose(file);
}

/**
 * This simply converts a string to a double.
 * It is meant to be ea first class function for loadMatrix()
 * 
 */

double returnDouble(std::string val, std::unordered_map<std::string, double>& myMap) {
    return atof(val.c_str());
}

/**
 * This function is used to classify qualitative data, like survey data
 * that lists "very likely, likely, neutral, unlikely, very unlikely"
 * or it can be used for data like movie tags.
 */

double classifyDouble(std::string val, std::unordered_map<std::string, double>& myMap) {
    if (myMap.find(val) != myMap.end()) {
        return myMap[val];
    }
    else {
        myMap.insert(std::make_pair(val, myMap.size()));
        return static_cast<double>(myMap[val]);
    }
}

/**
 * The main benchmarking function for checking the size of CSC and IVSparse matrices
 * 
 */

void benchmark(char* filepath, std::function<double(std::string, std::unordered_map<std::string, double>&)> func) {
    std::vector<std::tuple<uint, uint, double>> data;

    // loadMatrix(data, func, filepath); // for reading COO matrices (no metadata, just 3 values per row seperated by commas)
    std::cout << "Loading matrix... " << filepath << std::endl;
    load_mm_matrix(data, filepath);
    // generateMatrix(data, 500000, 10000, 99, 1, 1); //optional for generating a matrix
    std::cout << "Done loading matrix" << std::endl;
    data.resize(data.size());

    // construct a dictionary of the unique values
    std::map<uint, uint> uniqueValues;

    for (uint i = 0; i < data.size(); i++) {
        uniqueValues.insert(std::pair<uint, uint>(std::get<2>(data.at(i)), 0));
    }

    // print the number of unique values
    std::cout << "Number of unique values: " << uniqueValues.size() << std::endl;

    // for reading how many columns and rows are in the dataset
    uint cols = (uint)[&data] {
        int max = 0;
        for (uint i = 0; i < data.size(); i++) {
            if (std::get<1>(data.at(i)) > max) { // The value parameter may need to be changed depending on the data set.
                max = std::get<1>(data.at(i));
            }
        }
        return max + 1;
        }
    ();

    int rows = (uint)[&data] {
        int max = 0;
        for (uint i = 0; i < data.size(); i++) {
            if (std::get<0>(data.at(i)) > max) { // The value parameter may need to be changed depending on the data set.
                max = std::get<0>(data.at(i));  
            }
        }
        return max;
        }
    ();

    uint size = data.size();
    double density = (double)(static_cast<double>(size) / (double)(static_cast<double>(rows) * static_cast<double>(cols)));
    std::cout << "Rows: " << rows << std::endl;
    std::cout << "Cols: " << cols << std::endl;
    std::cout << "Size: " << size << std::endl;
    std::cout << "Density: " << density << std::endl;
    std::cout << "gigabytes: " << (double)(static_cast<double>(size) * 16) / (double)(1024 * 1024 * 1024) << std::endl;

    std::cout << "//////////////////////////////////////////////////////////// In: " << filepath << " ////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "CSC: " << std::endl;
    uint64_t csf1Size = buildMatrix<double, uint, 1>(data, rows, cols, size);
    std::cout << "VCSC: " << std::endl;
    uint64_t csf2Size = buildMatrix<double, uint, 2>(data, rows, cols, size);
    std::cout << "IVCSC: " << std::endl;
    uint64_t csf3Size = buildMatrix<double, uint, 3>(data, rows, cols, size);

    std::cout << "CSC: " << csf1Size << std::endl;
    std::cout << "CSF2: " << csf2Size << std::endl;
    std::cout << "IVCSC: " << csf3Size << std::endl;

    std::cout << std::endl;
    std::cout << "Ratios:" << std::endl;

    std::cout << "CSC " << (double)csf1Size / (size * 16) << std::endl;
    std::cout << "VCSC: " << (double)csf2Size / (size * 16) << std::endl;
    std::cout << "IVCSC: " << (double)csf3Size / (size * 16) << std::endl;
}

/**
 * Computes the average redundancy of a matrix. This method can be found in 
 * other benchmarking files in this folder.
 */

template <typename T, typename indexType, int compressionLevel>
void averageRedundancy(IVSparse::SparseMatrix<T, indexType, compressionLevel>& matrix) {
    const int numRows = matrix.rows();
    const int numCols = matrix.cols();
    std::cout << "Fetching redundancy" << std::endl;
    int colsWithValues = 0;
    double totalRedundancy = 0.0;
    std::unordered_map<double, double> uniqueValues_overall;


    for (int j = 0; j < numCols; ++j) {
        double totalValues = 0;
        double redundancy = 0;
        std::unordered_map<double, double> uniqueValues;

        for (typename IVSparse::SparseMatrix<T, indexType, compressionLevel>::InnerIterator it(matrix, j); it; ++it) {
            uniqueValues.insert(std::pair<double, int>(it.value(), 0));
            uniqueValues_overall.insert(std::pair<double, int>(it.value(), 0));
            totalValues++;
        }

        if (totalValues == 0 || uniqueValues.size() == 0)
            continue;
        else if (uniqueValues.size() == 1)
            redundancy = 1;
        else
            redundancy = 1 - (uniqueValues.size() / totalValues);
        totalRedundancy += redundancy;
        colsWithValues++;
    }
    std::cout << "Unique values in whole matrix: " << uniqueValues_overall.size() << std::endl;

    std::cout << "Avg Redundancy: " << totalRedundancy / static_cast<double>(colsWithValues) << std::endl;
}

// https://math.nist.gov/MatrixMarket/mmio-c.html
//
// https://math.nist.gov/MatrixMarket/mmio/c/example_read.c
// This code is slightly modified from the code found at the above link
void load_mm_matrix(std::vector<std::tuple<uint, uint, double>>& data, char* filename) {
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

    // Reads the dimensions and number of nonzeros
    if ((retCode = mm_read_mtx_crd_size(f, &rows, &cols, &nonzeros)) != 0) {
        std::cout << "\033[31;1;4mError: Could not read matrix dimensions.\033[0m" << std::endl;
        exit(1);
    }

    // Allocate memory for the matrix
    I = (int*)malloc(sizeof(uint));
    J = (int*)malloc(sizeof(uint));
    val = (double*)malloc(sizeof(double));
    data.reserve(nonzeros);
    // Read the matrix
    std::cout << "Done allocating memory" << std::endl;
    for (i = 0; i < nonzeros; i++) {
        fscanf(f, "%d %d %lg\n", &I[0], &J[0], &val[0]);
        data.push_back(std::make_tuple(I[0]--, J[0]--, val[0]));
    }

    // Close the file
    if (f != stdin)
        fclose(f);

    // Free the memory
    free(I);
    free(J);
    free(val);

    std::cout << "Loaded: " << filename << std::endl;

    return;
}

void generateMatrix(std::vector<std::tuple<uint, uint, double>>& data, int numRows, int numCols, int sparsity, uint64_t seed, uint64_t maxValue) {
    rng randMatrixGen = rng(seed);
    data.reserve(numRows * 2);
    std::unordered_map<int, int> uniqueValues;


    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            if (randMatrixGen.draw<int>(i, j, sparsity)) {
                int size = uniqueValues.size();
                int newVal = rand();
                uniqueValues.insert(std::pair<double, int>(newVal, 0));

                while (uniqueValues.size() == size) {
                    newVal = rand();
                    uniqueValues.insert(std::pair<double, int>(newVal, 0));
                }
                data.push_back(std::make_tuple(i, j, newVal));
                // data.push_back(std::make_tuple(i, j, 1));

            }
        }
    }
}


/**
 * Used to construct a matrix and check redundancy for IVCSC. This function was created to help
 * house matrices that were too big to store others, so we construct in here and let the system
 * destruct the matrix when it goes out of scope.
 */

template <typename T, typename indexType, int compressionLevel>
uint64_t buildMatrix(std::vector<std::tuple<uint, uint, double>>& data, uint rows, uint cols, uint nnz) {

    IVSparse::SparseMatrix<T, indexType, compressionLevel> matrix(data, rows, cols, nnz);
    
    if constexpr (compressionLevel == 3) {
        averageRedundancy<double, uint, 3>(matrix);
    }

    uint64_t size = matrix.byteSize();
    return size;
}

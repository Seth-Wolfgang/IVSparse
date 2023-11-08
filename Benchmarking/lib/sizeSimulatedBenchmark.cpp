/**
 * @file sizeSimulatedBenchmark.cpp
 * @author Seth Wolfgang
 * @brief This program was meant for a directed way of generating and testing a matrix of specific size, density, and value range.
 * @date 2023-08-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "suiteSparseBenchmarkFunctions.h"
#include "../../misc/matrix_creator.cpp"

#define DENSITY 0.1
#define VALUE_TYPE double

template <typename T>
void sizeTest(int rows, int cols, std::vector<std::vector<double>>& data, int maxValue);
void printDataToFile(std::vector<std::vector<double>>& data);
void __attribute__((optimize("Ofast"))) generateMatrix(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int numRows, int numCols, uint64_t seed);
inline __attribute__((optimize("O2"))) void adjustValues(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int maxValue, int seed);

template <typename T, typename indexType, int compressionLevel>
double averageRedundancy(IVSparse::SparseMatrix<T, indexType, compressionLevel>& matrix);

int main(int argc, char** argv) {
    std::vector<std::vector<double>> data(1);

    std::cout << "Running with " << argv[1] << " rows and " << argv[2] << " columns" << std::endl;
    if (argc != 4) {
        argv = (char**)malloc(4 * sizeof(char*));
        argv[1] = (char*)malloc(15 * sizeof(char));
        argv[2] = (char*)malloc(15 * sizeof(char));
        argv[3] = (char*)malloc(15 * sizeof(char));
        argv[1] = "1000000";
        argv[2] = "1000";
        argv[3] = "1000";
    }

    // for (int redundancy = 1; redundancy <= 1000; redundancy++) {
    sizeTest<double>(atoi(argv[1]), atoi(argv[2]), data, atoi(argv[3]));
    // }

    printDataToFile(data);

    return 0;
}

void printDataToFile(std::vector<std::vector<double>>& data) {
    FILE* file;
    file = fopen("../simSizeData.csv", "a");
    double gigaByteSize = (double)data.at(0).at(0) / 1000000000.0;

    fprintf(file, "%s\n", "COO size,dimension,nonzeros,Sparsity,Redundancy,100, CSF1 Size,CSF2 Size,CSF3 Size, CSF1 Bytes, CSF2 Bytes, CSF3 Bytes");
    for (int i = 0; i < data.size(); i++) {
        fprintf(file, " & %.2lf & %lu\\times%lu & %lu & %lf & %lf & 100 & %lf & %lf & %lf & %lu & %lu & %lu \\ \n", gigaByteSize, (uint64_t)data.at(i).at(1), (uint64_t)data.at(i).at(2), (uint64_t)data.at(i).at(3), data.at(i).at(4), data.at(i).at(5), 
        data.at(i).at(6), data.at(i).at(7), data.at(i).at(8), (uint64_t)data.at(i).at(9), (uint64_t)data.at(i).at(10), (uint64_t)data.at(i).at(11));

    }

}

template <typename T>
void sizeTest(int rows, int cols, std::vector<std::vector<double>>& data, int maxValue) {
    int spot = 0;
    std::vector<std::tuple<int, int, T>>  matrixData;
    generateMatrix(matrixData, rows, cols, 1);
    adjustValues(matrixData, maxValue, maxValue);


    IVSparse::SparseMatrix<T, INDEX_TYPE, 1> csf1(matrixData, rows, cols, matrixData.size());
    IVSparse::SparseMatrix<T, INDEX_TYPE, 2> csf2(matrixData, rows, cols, matrixData.size());
    IVSparse::SparseMatrix<T, INDEX_TYPE, 3> csf3(matrixData, rows, cols, matrixData.size());
    uint64_t cooSize = (sizeof(VALUE_TYPE) + sizeof(int) * 2) * matrixData.size();

    data.at(spot).push_back(cooSize);
    data.at(spot).push_back(rows);
    data.at(spot).push_back(cols);
    data.at(spot).push_back(matrixData.size());
    data.at(spot).push_back(100 * DENSITY);
    data.at(spot).push_back(100 * averageRedundancy<VALUE_TYPE, INDEX_TYPE, 2>(csf2));
    data.at(spot).push_back(100 * ((double)csf1.byteSize() / (double)cooSize));
    data.at(spot).push_back(100 * ((double)csf2.byteSize() / (double)cooSize));
    data.at(spot).push_back(100 * ((double)csf3.byteSize() / (double)cooSize));
    data.at(spot).push_back(csf1.byteSize());
    data.at(spot).push_back(csf2.byteSize());
    data.at(spot).push_back(csf3.byteSize());
}

void __attribute__((optimize("Ofast"))) generateMatrix(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int numRows, int numCols, uint64_t seed) {
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

inline __attribute__((optimize("O2"))) void adjustValues(std::vector<std::tuple<int, int, VALUE_TYPE>>& data, int maxValue, int seed) {
    srand(seed);

    #pragma omp parallel for
    for (uint64_t i = 0; i < data.size(); i++) {
        std::get<2>(data.at(i)) = rand() % maxValue + 1;
    }
}

template <typename T, typename indexType, int compressionLevel>
double averageRedundancy(IVSparse::SparseMatrix<T, indexType, compressionLevel>& matrix) {
    const int numRows = matrix.rows();
    const int numCols = matrix.cols();
    int colsWithValues = 0;
    double totalRedundancy = 0.0;

    #pragma omp parallel for reduction(+:totalRedundancy, colsWithValues)
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
            redundancy = 1;
        else
            redundancy = 1 - (uniqueValues.size() / totalValues);
        totalRedundancy += redundancy;
        colsWithValues++;
    }

    return totalRedundancy / static_cast<double>(colsWithValues);
}
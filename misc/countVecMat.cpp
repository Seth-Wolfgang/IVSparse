#include <iostream>
#include "IVSparse/SparseMatrix"
#include <fstream>

using namespace std;

template <typename T, typename indexType, int compressionLevel>
void averageRedundancy(IVSparse::SparseMatrix<T, indexType, compressionLevel> &matrix);

int main()
{

    int rows = 46985;
    int cols = 124836;
    int nnz = 5410236;

    uint32_t *vals = new uint32_t[nnz];
    uint32_t *indices = new uint32_t[nnz];
    uint32_t *colPointers = new uint32_t[cols + 1];

    // read in data from file and store in vals, indices, colPointers

    fstream file("data/X_data.txt");

    int count = 0;

    for (string line; getline(file, line);)
    {

        int value = stoi(line);
        vals[count] = value;
        count++;
    }

    file.close();

    file.open("data/X_indices.txt");

    count = 0;

    for (string line; getline(file, line);)
    {

        int value = stoi(line);
        indices[count] = value;
        count++;
    }

    file.close();

    file.open("data/X_indptr.txt");

    count = 0;

    for (string line; getline(file, line);)
    {

        int value = stoi(line);
        colPointers[count] = value;
        count++;
    }

    file.close();

    // create sparse matrix
    IVSparse::SparseMatrix<uint32_t, uint32_t, 1> X1(vals, indices, colPointers, rows, cols, nnz);
    IVSparse::SparseMatrix<uint32_t, uint32_t, 2> X2(vals, indices, colPointers, rows, cols, nnz);
    IVSparse::SparseMatrix<uint32_t, uint32_t, 3> X3(vals, indices, colPointers, rows, cols, nnz);

    // print the sparse matrix byte size
    cout << "CSC Size: " << X1.byteSize() << endl;
    cout << "VCSC Size: " << X2.byteSize() << endl;
    cout << "IVCSC Size: " << X3.byteSize() << endl;

    std::cout << "Ratios: " << std::endl;
    std::cout << "VCSC: " << (double)((double)X2.byteSize() / X1.byteSize()) << std::endl;
    std::cout << "IVCSC: " << (double)((double)X3.byteSize() / X1.byteSize()) << std::endl;

    std::cout << "rows: " << rows << std::endl;
    std::cout << "cols: " << cols << std::endl;
    std::cout << "nnz: " << nnz << std::endl;

    averageRedundancy<uint32_t, uint32_t, 3>(X3);
    std::cout << "density: " << (double)((double)nnz / (rows * cols)) << std::endl;

    // free
    delete[] vals;
    delete[] indices;
    delete[] colPointers;

    return 0;
}

template <typename T, typename indexType, int compressionLevel>
void averageRedundancy(IVSparse::SparseMatrix<T, indexType, compressionLevel> &matrix)
{
    const int numRows = matrix.rows();
    const int numCols = matrix.cols();
    int colsWithValues = 0;
    double totalRedundancy = 0.0;

    for (int j = 0; j < numCols; ++j)
    {
        double totalValues = 0;
        std::unordered_map<double, double> uniqueValues;

        for (typename IVSparse::SparseMatrix<T, indexType, compressionLevel>::InnerIterator it(matrix, j); it; ++it)
        {
            uniqueValues.insert(std::pair<double, int>(it.value(), 0));
            totalValues++;
        }
        if (totalValues == 0 || uniqueValues.size() == 0)
        {
            continue;
        }
        colsWithValues++;
        double redundancy = (uniqueValues.size() / totalValues);
        totalRedundancy += redundancy;
    }

    std::cout << "Avg Redundancy: " << totalRedundancy / static_cast<double>(colsWithValues) << std::endl;
}

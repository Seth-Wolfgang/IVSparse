#include <iostream>
#include "IVSparse/SparseMatrix"
#include <fstream>
#include <vector>
#include <string>

using namespace std;

int main()
{

    int num_rows = 32738;
    int num_cols = 2700;
    int nnz = 2031;

    vector<uint32_t> vals(nnz);
    vector<uint32_t> rows(nnz);
    vector<uint32_t> cols(nnz);

    fstream file("data/pbmc3k_csc/data.txt");
    for (string line; getline(file, line);)
    {
        int value = stoi(line);
        vals.push_back(value);
    }
    file.close();

    file.open("data/pbmc3k_csc/indices.txt");
    for (string line; getline(file, line);)
    {
        int value = stoi(line) - 1;
        rows.push_back(value);
    }
    file.close();

    file.open("data/pbmc3k_csc/indptr.txt");
    for (string line; getline(file, line);)
    {
        int value = stoi(line) - 1;
        cols.push_back(value);
    }
    file.close();

    // get a vector of tuples of (row, col, val)
    vector<tuple<uint32_t, uint32_t, uint32_t>> tuples(nnz);

    for (int i = 0; i < nnz; i++)
    {
        tuples[i] = make_tuple(rows[i], cols[i], vals[i]);
    }

    // make a sparse matrix
    IVSparse::SparseMatrix<uint32_t, uint32_t> X(tuples, num_rows, num_cols, nnz);

    std::cout << "Size: " << X.byteSize() << " bytes" << std::endl;
    std::cout << "Number of non-zero elements: " << X.nonZeros() << std::endl;
    std::cout << "Number of rows: " << X.rows() << std::endl;
    std::cout << "Number of columns: " << X.cols() << std::endl;

    IVSparse::SparseMatrix<uint32_t, uint32_t, 2> X2(X);
    IVSparse::SparseMatrix<uint32_t, uint32_t, 1> X3(X);

    // print the sparse matrix byte size
    std::cout << "CSC Size: " << X3.byteSize() << std::endl;
    std::cout << "VCSC Size: " << X2.byteSize() << std::endl;

    return 0;
}
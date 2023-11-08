@page conversion_example Conversion Examples
@tableofcontents

@section conversion Conversions

There are many ways to convert between compression formats in IVSparse. This page will show you how to convert between the different formats.

@subsection ivsparse_conversions IVSparse Conversions

For the three conversion formats supported by IVSparse (CSC, VCSC, and IVCSC) its simple to convert between each of these. There isn't in place conversion but you can easily create a new matrix for the desired format and pass in the old matrix to the constructor or use a method call. Here are some examples:

```cpp
// exampleMatrix is a IVCSC matrix (compression level 3)
IVSparse::SparseMatrix<double> exampleMatrix(values, rowIndices, colIndices, numRows, numCols, numNonZeros);

// convert exampleMatrix to a CSC matrix (compression level 1)

// method 1
IVSparse::SparseMatrix<double, uint64_t, 1> cscMatrix(exampleMatrix);

// method 2
IVSparse::SparseMatrix<double, uint64_t, 1> cscMatrix = exampleMatrix.toCSC();
```

The `toCSC()` method is exlusive to the non-CSC compression formats as each compression format has two methods for converting to the other two compression formats. This is demonstrated below:

* CSC - `toVCSC()` and `toIVCSC()`
* VCSC - `toCSC()` and `toIVCSC()`
* IVCSC - `toCSC()` and `toVCSC()`

@subsection eigen_conversion Eigen Conversions

The other conversion supported by IVSparse is to convert an IVSparse Matrix to an Eigen Sparse Matrix. This is also very simple to do. Here are some examples:

```cpp
// example matrix
IVSparse::SparseMatrix<double> exampleMatrix(values, rowIndices, colIndices, numRows, numCols, numNonZeros);

// convert exampleMatrix to an Eigen Sparse Matrix
Eigen::SparseMatrix<double> eigenMatrix = exampleMatrix.toEigen();
```
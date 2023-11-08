@page construction_example Construction Examples
@tableofcontents

@section constructors Constructors

These are the constructors offered by the different formats.

* Default Constructor - Creates an empty matrix
* Eigen Constructor - Creates a matrix from an Eigen Sparse Matrix
* COO Constructor - Creates a matrix from COO data
* CSC Constructor - Creates a matrix from CSC data
* Copy Constructor - Creates a matrix from another matrix
* Conversion Constructor - Creates a matrix from another format
* Vector Constructor - Creates a matrix from a vector
* Array of Vectors Constructor - Creates a matrix from an array of vectors
* File Constructor - Creates a matrix from a file

@subsection default Default Constructor

The default constructor has no use other than to create an empty matrix. Since it is not possible to insert values into an IVSparse Matrix after construction this is only useful for creating a matrix to be overwritten later. It's also worth mentioning that IVCSC also has a construtor that allows for the specification of rows and columns for an empty matrix. This is essentially the same as the default constructor but allows for the specification of rows and columns.

```cpp
IVSparse::SparseMatrix<double> matrix();
```

@section outside_formats Loading Data into IVSparse

This is how to load data into IVSparse from outside sources using COO, CSC, or Eigen.

@subsection coo COO Input Data

First is loading data in using COO which is a very common format for sparse data such as Matrix Market. IVSparse takes in a vector of tuples where each tuple is (row, column, value). The vector doesn't need to be sorted but does need to contian no duplicates.

```cpp
std::vector<std::tuple<uint32_t, uint32_t, double>> cooData = {
    {0, 0, 1},
    {1, 1, 2},
    {2, 2, 3},
    {3, 3, 4}
};

int rows = 4;
int cols = 4;
int nnz = 4;

IVSparse::SparseMatrix<double> matrix(cooData, rows, cols, nnz);
```

@subsection raw_csc Raw CSC Input Data

It's also easy for IVSparse to use raw CSC data to construct matrices and is often used as well for sparse applications. IVSparse takes in pointers to the three arrays for CSC/CSR and the metadata needed about the matrix. The arrays do need to follow CSC format conventions and given the correct storage order. For example if using CSC data make a column major matrix, if using CSR data make a row major matrix. An example for using raw CSC pointers is shown below:

```cpp
double values[4] = {1, 2, 3, 4};
uint32_t rowIndices[4] = {0, 1, 2, 3};
uint32_t outerPointers[5] = {0, 1, 2, 3, 4};

int rows = 4;
int cols = 4;
int nnz = 4;

IVSparse::SparseMatrix<double> matrix(values, rowIndices, outerPointers, rows, cols, nnz);
```

@subsection eigen Eigen Sparse Matrix Input Data

The easiest way to make an IVSparse Matrix is to construct one from an already existing Eigen sparse matrix. Eigen sparse matrices are just CSC matrices so the process is the same as the CSC example above but with far less parameters for the programmer to manage. IVSparse can also take either column or row major Eigen sparse matrices. An example of this is shown below:

```cpp
Eigen::SparseMatrix<double> eigenMatrix(4, 4);
eigenMatrix.insert(0, 0) = 1;
eigenMatrix.insert(1, 1) = 2;
eigenMatrix.insert(2, 2) = 3;
eigenMatrix.insert(3, 3) = 4;

IVSparse::SparseMatrix<double> matrix(eigenMatrix);
```

@section moving_data Moving Data in IVSparse

There are many ways to move data around in IVSparse once in one of the three formats supported. The main ways are by using the copy constructor, the conversion constructor, the assignment operator, the file constructor, or the vector constructors.


@subsection file File Constructor

Constructing from a file is one of the easiest ways to move data around. Simply save a matrix on disk using the `.write()` method and simply load it in with the same template parameters as the matrix saved. An example of this is shown below:

```cpp
IVSparse::SparseMatrix<double> exampleMatrix(values, rowIndices, outerPointers, rows, cols, nnz);
exampleMatrix.write();

IVSparse::SparseMatrix<double> matrix("exampleMatrix.ivs");
```

@subsection conversion Conversion/Copy Constructors

It's simple to copy or move between compression levels in IVSparse. To deep copy a matrix just use one matrix as the parameter to the constructor of another matrix of the same template parameters. To convert between compression levels do the same thing except the compression level parameter can be different and the constructor will do the conversion itself. An example of this is shown below:

```cpp
IVSparse::SparseMatrix<double> exampleMatrix(values, rowIndices, outerPointers, rows, cols, nnz);

// Copies exampleMatrix
IVSparse::SparseMatrix<double> copyMatrix(exampleMatrix);

// Converts exampleMatrix to VCSC and then copies it
IVSparse::SparseMatrix<double, uint64_t, 2> conversionMatrix(exampleMatrix);
```

@subsection vector_constructors Vector Constructors

The last primary way to move data in IVSparse is through the use of vectors. There are two vector constructors to make an IVSparse Matrix, the single vector constructor and the array of vectors constructor. The single vector constructor simply turns a vector into a matrix with a single row or column. The array of vectors constructor takes in an array of vectors and turns it into a matrix. An example of this is shown below:

```cpp
IVSparse::SparseMatrix<double> exampleMatrix(values, rowIndices, outerPointers, rows, cols, nnz);

// get vectors from exampleMatrix
IVSparse::SparseMatrix<double>::Vector vector1(exampleMatrix, 0);
IVSparse::SparseMatrix<double>::Vector vector2(exampleMatrix, 1);
IVSparse::SparseMatrix<double>::Vector vector3(exampleMatrix, 2);
IVSparse::SparseMatrix<double>::Vector vector4(exampleMatrix, 3);

// Make a matrix from a single vector
IVSparse::SparseMatrix<double> singleVectorMatrix(vector1);

// Make a matrix from an array of vectors
std::vector<IVSparse::SparseMatrix<double>::Vector> vectors = {vector1, vector2, vector3, vector4};
IVSparse::SparseMatrix<double> arrayOfVectorsMatrix(vectors);
```
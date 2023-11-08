@page matrix_getters Matrix Getters
@tableofcontents

@section getters Getters

There are a lot of methods that give information about the matrix. Most of these are shared between all three supported formats but a few are specific to the data structure of the format. These are the properties that are shared between all three formats which have getter methods.

* Rows - Returns the number of rows in the matrix
* Columns - Returns the number of columns in the matrix
* Inner Dimension - Returns the inner dimension of the matrix (against storage order)
* Outer Dimension - Returns the outer dimension of the matrix (with storage order)
* Non-Zeros - Returns the number of non-zeros in the matrix
* Byte Size - Returns the number of bytes the matrix takes up in memory
* Storage Order - Returns the storage order of the matrix
* Coeff - Returns the value at a given row and column

Examples of these are shown below:

```cpp
IVSparse::SparseMatrix<double> matrix(values, rowIndices, outerPointers, rows, cols, nnz);

matrix.print();

std::cout << "Rows: " << matrix.rows() << std::endl;
std::cout << "Columns: " << matrix.cols() << std::endl;
std::cout << "Inner Dimension: " << matrix.innerSize() << std::endl;
std::cout << "Outer Dimension: " << matrix.outerSize() << std::endl;
std::cout << "Non-Zeros: " << matrix.nonZeros() << std::endl;
std::cout << "Byte Size: " << matrix.byteSize() << std::endl;
std::cout << "Is Column Major: " << matrix.isColumnMajor() << std::endl;
std::cout << "coeff at (0, 0): " << matrix.coeff(0, 0) << std::endl;
```

The output for this program is:

```
IVSparse Matrix:

1 0 0 0
0 2 0 0
0 0 3 0
0 0 0 4

Rows: 4
Columns: 4
Inner Dimension: 4
Outer Dimension: 4
Non-Zeros: 4
Byte Size: 132
Is Column Major: 1
coeff at (0, 0): 1
```

@section csc CSC Getters

The only special properties of CSC are the access methods to the underlying pointers of CSC. These pointers are the same that Eigen also has underneath their Sparse Matrix class and are simple pointer arrays to the relevant data. That means there are three methods that are specific to CSC which are:

* Values - Returns a pointer to the values array
* Inner Indices - Returns a pointer to the inner indices array
* Outer Pointers - Returns a pointer to the outer pointers array

Examples of these are shown below:

```cpp
// Same matrix as above
IVSparse::SparseMatrix<double, uint64_t, 1> matrix(values, rowIndices, outerPointers, rows, cols, nnz);

double* valuesPtr = matrix.getValues();
uint32_t* innerIndicesPtr = matrix.getInnerIndices();
uint32_t* outerPointersPtr = matrix.getOuterPointers();

for (int i = 0; i < matrix.nonZeros(); i++) {
    std::cout << valuesPtr[i] << " ";
}
```

The output for this program is:

```
1 2 3 4
```

@section vcsc VCSC Getters

Since VCSC is a derivative of CSC the specific getters for VCSC are similar to CSC. The main difference is that the pointers are specific to individual columns instead of the entire matrix. That means one must specify which column/row they are wanting a pointer for. The other main difference is the contents of the arrays. The values array now only holds the unique values for the column, there is a counts which holds the number of occurences of the unique values, and the indices of the values in the column. The three arrays to know of are:

* Unique Values - Returns a pointer to the unique values array of a column
* Counts - Returns a pointer to the counts array of a column
* Inner Indices - Returns a pointer to the inner indices array of a column

The only issue is that one needs to know the size of the arrays in order to safely access them. For this there are two more methods now provided which are:

* Number of Unique Values - Returns the number of unique values in a column
* Number of Indices - Returns the number of indices in a column

Examples of these are shown below:

```cpp
// Same matrix as above
IVSparse::SparseMatrix<double, uint64_t, 2> matrix(values, rowIndices, outerPointers, rows, cols, nnz);

double* valuesPtr = matrix.getValues(0);
uint32_t* countsPtr = matrix.getCounts(0);
uint32_t* innerIndicesPtr = matrix.getIndices(0);

for (int i = 0; i < matrix.getNumUniqueVals(0); i++) {
    std::cout << valuesPtr[i] << " ";
}
```

The output for this program is:

```
1
```

@section ivcsc IVCSC Getters

IVCSC is somewhat different than the other supported compression formats. Similar to IVCSC the data for each column is stored and accessed separately but the difference is that IVCSC data is stored in a contiguous byte array as opposed to three specific data arrays. This means accessing the data is very difficult without the use of the iterator. So the only additional getters for IVCSC are:

* Vector Size - Returns the size of the vector for a column in bytes
* Vector Pointer - Returns a pointer to the vector for a column

Examples of these are shown below:

```cpp
// Same matrix as above
IVSparse::SparseMatrix<double, uint64_t, 3> matrix(values, rowIndices, outerPointers, rows, cols, nnz);

uint32_t vectorSize = matrix.getVectorSize(0);
void* vectorPtr = matrix.vectorPointer(0);
```

These methods have much less use then the other two formats as the nature of how the data is stored in IVCSC makes it inherently difficult to access the data without the use of the iterator. These can be safely ignored but there are edge cases where they can be useful such as wanting to look into the raw memory for confirmation or to know the size of an individual vector in bytes. 
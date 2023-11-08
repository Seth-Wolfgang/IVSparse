@page manipulation_examples Matrix Manipulation Examples
@tableofcontents

@section mat_manips Matrix Manipulations

There are a few supported way to manipulate IVSparse Matrices. There are only a few because IVSparse is mostly a read-only format that is difficult to traverse and manipulate in place. For these reasons the methods for manipulating sparse matrices supported by IVSparse are as follows:

* Get Vector
* Transpose
* Append
* Slice

@subsection get_vector Get Vector

The get vector method is a simple method that returns an IVSparse vector that is a copy of the column specified in a matrix. This is one of the most useful methods for manipulating IVSparse matrices as it allows you to pull and use a single column of a matrix at a time. This is the same as constructing a vector by passing in an IVSparse matrix and a column number to the constructor. Here is an example of how to get a vector from a matrix. 

```cpp
// example matrix
IVSparse::SparseMatrix<double> exampleMatrix(values, rowIndices, colIndices, numRows, numCols, numNonZeros);

// get the first column of exampleMatrix
IVSparse::SparseMatrix<double>::Vector firstColumn = exampleMatrix.getVector(0);

// print the first column
firstColumn.print();
```

The above code will print out the following:

```
1 0 0 0
```

@subsection transpose Transpose

There are two methods that transpose, there is the `.transpose()` method which returns a copy of the called matrix transposed, and there is the `.transposeInPlace()` method which transposes the called matrix in place. Here are some examples:

```cpp
// example matrix
IVSparse::SparseMatrix<double> exampleMatrix(values, rowIndices, colIndices, numRows, numCols, numNonZeros);

// transpose exampleMatrix
IVSparse::SparseMatrix<double> transposedMatrix = exampleMatrix.transpose();

// transpose exampleMatrix in place
exampleMatrix.transposeInPlace();
```

@subsection append Append

Append is a method that allows you to append a vector onto the end of a matrix. This doesn't allow you to append matrices together or append vectors against the storage order. Combined with the ability to transpose matrices and pull out slices of matrices you can manipulate matrices in many ways. Here are some examples:

```cpp
// example matrix
IVSparse::SparseMatrix<double> exampleMatrix(values, rowIndices, colIndices, numRows, numCols, numNonZeros);

// example vector
IVSparse::SparseMatrix<double>::Vector exampleVector(exampleMatrix, 0);

// append exampleVector to exampleMatrix
exampleMatrix.append(exampleVector);

exampleMatrix.print();
```

The above code will print out the following:

```
IVSparse Matrix:

1 0 0 0 1
0 2 0 0 0
0 0 3 0 0
0 0 0 4 0
```

@subsection slice Slice

Slice is just a simple way of getting a submatrix of a matrix in the form of an array of vectors. This can be put right back into a matrix constructor to create a new matrix. This is helpful for getting only the middle chunk of a matrix or for splitting a matrix into smaller pieces. Slice also doesn't alter the original matrix. Here are some examples:

```cpp
// example matrix
IVSparse::SparseMatrix<double> exampleMatrix(values, rowIndices, colIndices, numRows, numCols, numNonZeros);

// get a slice of exampleMatrix
std::vector<IVSparse::SparseMatrix<double>::Vector> slice = exampleMatrix.slice(1, 3);

// print the slice
for (auto vector : slice) {
    vector.print();
}
```

The above code will print out the following:

```
0 2 0 0
0 0 3 0
```
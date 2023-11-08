@page iterators_and_vectors Iterators and Vectors
@tableofcontents

@section iterators_and_vectors Iterators and Vectors

There are two subclasses of IVSparse Matrices which are Vectors and InnerIterators. The vector class is used to help interact with IVSparse Matrices and the InnerIterator class is used for simple traversal of IVSparse Matrices. This page will show the basics of how to use these two classes.

@subsection vector Vector

There are two main ways to construct a IVSparse Vector. The first is use the Vector constructor by passing in a matrix or another vector. The second is to get the vector from a matrix method call and assign it to a vector. These two ways do not include the ability to make a vector from raw data. Vectors can only be constructed from already compressed IVSparse data. Here are some examples:

```cpp
// example matrix
IVSparse::SparseMatrix<double> exampleMatrix(values, rowIndices, colIndices, numRows, numCols, numNonZeros);

// example vector
IVSparse::SparseMatrix<double>::Vector exampleVector(exampleMatrix, 0);

// get the first column of exampleMatrix
IVSparse::SparseMatrix<double>::Vector firstColumn = exampleMatrix.getVector(0);
```

Once in possession of a vector the getters are mainly the same as a matrix. The following are the primary getter methods for vectors:

* Coeff - returns the value at the specified index
* Length - returns the length of the vector
* NonZeros - returns the number of non-zero values in the vector

Finally, vectors are tied to their compression level. Much like the matrices, the data of the vector is stored in the same format as the compression level specified meaning vectors only work with matrices and iterators of the same compression level. This is important to remember when working with vectors.

@subsection inner_iterator InnerIterator

The InnerIterator class is used for traversal of IVSparse Matrices. They are all forward traversal iterators that only iterate over storage order of the matrix. This means that for a column major matrix the iterator will iterate over a single column and to iterate over the entire matrix an iterator needs to be made for each column. The iterators for VCSC and IVCSC as well iterate in order of values and not indices. This breaks most helpful ordering and makes traversals for some operations like SpMM slower without index ordering. There are two containers which can be traversed, matrices and vectors. Some examples of how to use the InnerIterator class are below:

```cpp
// example matrix
IVSparse::SparseMatrix<double> exampleMatrix(values, rowIndices, colIndices, numRows, numCols, numNonZeros);

// example vector
IVSparse::SparseMatrix<double>::Vector exampleVector(exampleMatrix, 0);

// iterate over exampleMatrix column 0
for (IVSparse::SparseMatrix<double>::InnerIterator it(exampleMatrix, 0); it; ++it) {
    // do something with the iterator
}

// iterate over exampleVector
for (IVSparse::SparseMatrix<double>::InnerIterator it(exampleVector); it; ++it) {
    // do something with the iterator
}
```

Once traversing a matrix or vector there are a few methods to get the information from the iterator. The following are the primary getter methods for iterators:

* Get Index - returns the index of the current value
* Outer Dimension - returns the outer dimension of the matrix being iterated over
* Row - returns the row of the current value
* Col - returns the column of the current value
* Value - returns the value of the iterator
* Coeff - changes the current value of the iterator

The `.coeff(T newValue)` method is the only way to change individual values of a matrix or vector. The only issue with changing values using this is that it changes the value tied to potentially multiple indices. This means that at least for VCSC and IVCSC the only way to change individual values is to change every instance of a unique value to another value. This is not recommended for use by those who are unsure of how this may cause the matrix to change or break. If one needs to change individual values VCSC and IVCSC matrices should be converted to a format that allows for this such as CSC. 

The other getter methods are fairly self explanitory, with the most useful ones being the index and value getters. 

Following are some examples of how to traverse a matrix and print out the matrix data:

```cpp
// example matrix
IVSparse::SparseMatrix<double> exampleMatrix(values, rowIndices, colIndices, numRows, numCols, numNonZeros);

exampleMatrix.print();

// iterate over the entire matrix
for (uint32_t i = 0; i < exampleMatrix.outerSize(); i++) {
    for (typename IVSparse::SparseMatrix<double>::InnerIterator it(exampleMatrix, i); it; ++it) {
        std::cout << "(" << it.row() << ", " << it.col() << ") = " << it.value() << std::endl;
    }
}
```

The above code will print out the following:

```
IVSparse Matrix:

1 0 0 0
0 2 0 0
0 0 3 0
0 0 0 4

(0, 0) = 1
(1, 1) = 2
(2, 2) = 3
(3, 3) = 4
```
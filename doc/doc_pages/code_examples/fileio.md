@page file_io_example File I/O Example
@tableofcontents

@section file_io File I/O

IVSparse supports reading and writing to files for its three compression formats it supports. By writing the matrix to disk using the `.write()` method you can then read it back in using the corresponding constructor and using the correct template parameters. Here are some examples:

```cpp
// example matrix
IVSparse::SparseMatrix<double> exampleMatrix(values, rowIndices, colIndices, numRows, numCols, numNonZeros);

// write exampleMatrix to a file
exampleMatrix.write("exampleMatrix.ivsparse");

// read exampleMatrix from a file
IVSparse::SparseMatrix<double> exampleMatrixFromFile("exampleMatrix.ivsparse");
```

The `.write()` method will write the matrix to disk in the compression format it was currently in when the method was called but the constructor will attempt to read in the compression level of the template paramter so it's important that you specify the correct compression level in the template parameters. This is also a good way to split up a large matrix and work with smaller pieces of it. By writting parts of a matrix to file you can read them back in individually and work with them.
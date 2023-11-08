@page getting_started Getting Started
@tableofcontents

This page serves as the main entry point for those looking to jump into coding with the IVSparse Library. 
This page will go through how to get IVSparse, how to compile it, how to use it, and where to learn more about it.

@section getting_ivsparse Getting IVSparse

To get IVSparse you need to go to our Github Repository and download the source code.

@subsection install_ivsparse Installing IVSparse

The process for installing IVSparse is similar to the process for isntalling the Eigen library. As a header library one only needs to include the path to the IVSparse directory in their compiler's include path. This is shown in how to compile IVSparse below. This is easy to do by cloning the IVSparse repository into your project directory and including the path to the IVSparse directory in your compiler's include path. Since Eigen is a required dependency of IVSparse you will also need to include the path to the Eigen library in your compiler's include path. This is also shown in how to compile IVSparse below. IVSparse does not have CMake support yet.

@subsection program_ivsparse Programming with IVSparse

Below is a simple example of how to create a IVSparse SparseMatrix object and print it out.

```cpp
#include <Eigen/Sparse>
#include <IVSparse/SparseMatrix>
#include <iostream>

int main() {

    int rows = 4;
    int cols = 4;
    int nnz = 4;

    int values[4] = {1, 2, 3, 4};
    int rowIndices[4] = {0, 1, 2, 3};
    int outerPointers[5] = {0, 1, 2, 3, 4};

    IVSparse::SparseMatrix<int> matrix(values, rowIndices, outerPointers, rows, cols, nnz);

    matrix.print();
}
```

The output for this program is:

```cpp
IVSparse Matrix:

1 0 0 0
0 2 0 0
0 0 3 0
0 0 0 4
```

@subsection compiling_ivsparse Compiling IVSparse

To compile IVSparse you need to include the path to IVSparse and Eigen in your compiler's include path.

`g++ test.cpp -I /path/to/IVSparse -I /path/to/Eigen -o test`

To activate the parallel sections run with the `-fopenmp` flag. Running with the `-02` optimization flag is also recommended.

@subsubsection preprocessor_macros Preprocessor Macros

Parallelism is on by default when running with `-fopenmp` but can be turned off by defining `IVSPARSE_DONT_PARALLEL` before including IVSparse.

```cpp
#define IVSPARSE_DONT_PARALLEL
#include <IVSparse/SparseMatrix>
```

To turn off debugging flags define `IVSPARSE_DEBUG_OFF` before including IVSparse.

```cpp
#define IVSPARSE_DEBUG_OFF
#include <IVSparse/SparseMatrix>
```

@section mats_vecs_iters Matrices, Vectors, and Iterators

The three formats supported by IVSparse all have their specific matrix implementations but also have their own vector and iterator implementations as well. The only conversion between formats however happens on a matrix level. This means a IVCSC Vector can only be created from a IVCSC Matrix and not later turned directly into a CSC vector.

@subsection matrices Matrices

There are three matrix formats to keep track of. These are CSC, VCSC, and IVCSC. These stand for Compressed Sparse Column (CSC), Value Compressed Sparse Column (VCSC), and Index and Value Compressed Sparse Column (IVCSC). Each one has its own implementation of data storage.

@subsubsection templates Template Parameters

There are four template parameters to know of for IVSparse as shown below.

```cpp
#define VALUE_TYPE double
#define INDEX_TYPE uint64_t
#define COMPRESSION_LEVEL 3
#define COLUMN_MAJOR true

IVSparse::SparseMatrix<VALUE_TYPE, INDEX_TYPE, COMPRESSION_LEVEL, COLUMN_MAJOR> matrix;
```

The first template parameter is the value type of the matrix, which must be specified at compile time. Index type is the next one and is the type used to store indices of the matrix. The default value for index type is a `uint64_t` The third is the compression level desired. Level 1 is CSC, level 2 is VCSC, and level 3 is IVCSC. The default for this template paramter is compression level 3, IVCSC. The last template parameter is the major order of the matrix. This is a boolean value with true being column major and false being row major. The default value for this template parameter is true, column major. 

Since everything but the value type has a defiend default the above code can be simplified to:

```cpp
IVSparse::SparseMatrix<double> matrix;
```

@subsubsection levels IVSparse Compression Levels

There are currently three supported sparse matrix compression formats in IVSparse. These are CSC, VCSC, and IVCSC. These are all different ways of storing sparse data and have different advantages and disadvantages. The diagram below shows the differences between the different levels.

![IVSparse Levels Diagram](/doc/doc_pages/images/levels_fig.png)

*CSC*

CSC is used often in the storage and use of sparse data and is supported here to allow for easy interoperability with other libraries and to allow for easy transition between compression levels. CSC is also the fastest format to traverse and is the default format for Eigen Sparse Matrices.

*VCSC*

VCSC is a derivative of CSC meant to implement value compression on redundant values. This is done by storing only the unique values of each column/row and the number of occurances of each value. This means there are three arrays per outer dimension, the values, the counts, and the inner indices of the values. This format has fast scalar operations and is somewhat fast to traverse due to the counts array. This format also needs to have a fair amount of redundant values to benefit over CSC and is not recommended for data that is mostly unique.

*IVCSC*

IVCSC is the format meant to focus primarily on compressing. This is done by compressing both indices as well as values. Firstly, IVCSC stores each column in its own contiguous block of memory. Inside each column's memory is a series of runs. Runs are associated with each unique value in the column. Runs are sorted by unique value in ascending order. The format of a run is the value, the width of the follwing indices, the indices associated with the unique value positive delta encoded, and finally a delimiter to signify the end of a run. IVCSC uses both index and value compression to achieve the highest compression ratio. The values are compressed by only storing the unique values of each column and the indices for each individual run are positive delta encoded and then bytepacked into the smallest usable size. This format is the slowest to traverse but has the highest compression ratio, that isn't to say one can't do operations with IVCSC they just won't be quite as fast as a CSC matrix.

@subsection vectors Vectors

Vectors are a matrix with a single row or column. In IVSparse matrices can have a single row or column but are still considered matrices. For working specifically with vectors there is a vector sublclass provided. The vector subclass is meant to allow for manipulation of sparse matrices. Vectors are used to slice, append, and otherwise manipulate an otherwise read only format matrix. The vectors also are tied to their compression level and only work with matrices of the same compression level. Otherwise they work very similar to their matrix counterparts. Below an example of how to get a vector is shown. 

```cpp
IVSparse::SparseMatrix<double> matrix(values, rowIndices, outerPointers, rows, cols, nnz);

IVSparse::SparseMatrix<double>::Vector vector(matrix, 0);

vector.print();
```

The output for this program using the same matrix from above is:

```cpp
IVSparse Vector:

1 0 0 0
```

@subsection iterators Iterators

In IVSparse we have iterators made for each compression level to traverse their specific data structures. Each iterator is a forward traversal iterator meant to iterate over one column at a time. For this reason they are called inner iterators. So to traverse a matrix you need to iterate over each column with a new iterator. Iterators are also tied to their compression level and only work with matrices of the same compression level. It's also worth noting they work very similar to an Eigen inner iterator and they also can iterate over IVSparse vectors the same way. Below is an example of how to use an iterator.

```cpp
IVSparse::SparseMatrix<double> matrix(values, rowIndices, outerPointers, rows, cols, nnz);

int sum = 0;

// Traverse over the entire matrix
for (uint32_t i = 0; i < outerDim; ++i) {
    for (typename IVSparse::SparseMatrix<double>::InnerIterator it(matrix, i); it; ++it) {
        sum += it.value();
    }
}

std::cout << sum << std::endl;
```

Using the same diagonal matrix from previous examples sum would be equal to `1 + 2 + 3 + 4` which is `10`.

@section next Next Steps

Now that you have the basics down you can either start coding with IVSparse or you can dig deeper in the examples, reference guide, FAQ, or contact us directly. 

@subsection examples Examples

See our examples section to get more examples on how to use IVSparse and the compression formats it supports.

@subsection ref Reference Guide

Look at the reference guide to see all the functionality offered by IVSparse in more detail.

@subsection faq FAQ

If we end up making a FAQ this is where a link to it will be!

@subsection contact Contact

To contact us the best way is to leave a request on the Github repo but you can also email us at ruitersk@mail.gvsu.edu or at wolfgans@mail.gvsu.edu.


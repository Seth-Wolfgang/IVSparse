@page arithmetic_examples Arithmetic Examples
@tableofcontents

@section arithmetic Arithmetic Operations

There are many ways to perform arithmetic operations on IVSparse Matrices. This page will show you how to perform the basic arithmetic operations on IVSparse Matrices. These can be broken down into a few groups. These are BLAS routines, vector operations, and matrix operations. For dense matrices since IVSparse has no dense format the Eigen Linear Algebra Library is used for returns of dense matrices and vectors.

@subsection setup Setup

Firstly, to set up some examples we need to create some matrices and vectors. Here are some examples of how to do that:

```cpp
// IVSparse Matrix
IVSparse::SparseMatrix<double> exampleMatrix(values, rowIndices, colIndices, numRows, numCols, numNonZeros);

exampleMatrix.print();

// IVSparse Vector
IVSparse::SparseMatrix<double>::Vector exampleVector(exampleMatrix, 0);

std::cout << "IVSparse Vector: " << std::endl;
exampleVector.print();

// Eigen Sparse Matrix
Eigen::SparseMatrix<double> eigenMatrix = exampleMatrix.toEigen();

// Eigen Dense Matrix
Eigen::MatrixXi eigenDenseMatrix(5, 4);
eigenDenseMatrix << 0, 6, 4, 1,
                    3, 9, 0, 0,
                    0, 0, -5, -10,
                    9, 8, 7, 6,
                    0, 0, 0, 0;

std::cout << "Eigen Dense Matrix: " << std::endl;
std::cout << eigenDenseMatrix << std::endl;

// Eigen Dense Vector
Eigen::VectorXi eigenVector(5);
eigenVector << 1, 2, 3, 4, 5;

std::cout << "Eigen Dense Vector: " << std::endl;
std::cout << eigenVector << std::endl;
```

The above code will print out the following:

```
IVSparse Matrix:

4 0 3 0 2 
0 4 0 2 0 
4 0 3 0 1 
0 2 0 2 0 
3 0 4 0 3

IVSparse Vector:
4 0 4 0 3

Eigen Dense Matrix:
  0   6   4   1
  3   9   0   0
  0   0  -5 -10
  9   8   7   6
  0   0   0   0

Eigen Dense Vector:
1 
2 
3 
4 
5
```

@subsection blas BLAS Routines

For BLAS routines IVSparse supports the following:

* Scalar Multiplication (Level 1)
* SpMV (Level 2)
* SpMM (Level 3)

*Scalar Multiplication*

Starting with scalar multiplication there is both an in-place and new return version of this method. The in-place version is done with the `*=` operator and the new return version is done with the `*` operator. This operation goes through the entire matrix and multiplies each value. In VCSC this is very fast due to keeping only unique values for a column in contiguous memory. Here are some examples:

```cpp
IVSparse::SparseMatrix<double> scalarMultMatrix = exampleMatrix * 2;

scalarMultMatrix.print();

exampleMatrix *= 2;
exampleMatrix.print();
```

The above code will print out the following:

```
IVSparse Matrix:

8 0 6 0 4 
0 8 0 4 0 
8 0 6 0 2 
0 4 0 4 0 
6 0 8 0 6 

IVSparse Matrix:

8 0 6 0 4 
0 8 0 4 0 
8 0 6 0 2 
0 4 0 4 0 
6 0 8 0 6 
```

*SpMV*

For multiplying a IVSparse Matrix by a vector there are two supported ways of doing this, using Eigen vectors or IVSparse vectors. This is done in O(n^2) and is slower than the Eigen implementation of this routine. Since the multiplication process is the same for both the example will only use Eigen vectors for simplicity. Here are some examples:

```cpp
Eigen::VectorXi eigenResult = exampleMatrix * eigenVector;

std::cout << "Eigen Result: " << std::endl;
std::cout << eigenResult << std::endl;
```

The above code will print out the following:

```
Eigen Result:
23
16
18
12
30
```

*SpMM*

For multiplying an IVSparse Matrix by a dense matrix the only supported way is by using an Eigen matrix to multiply. This algorithm is done in O(n^3) and returns an Eigen::Matrix. Currently there is only support for Sparse * Dense operations. Here are some examples:

```cpp
Eigen::MatrixXi eigenMatrixResult = exampleMatrix * eigenDenseMatrix;

std::cout << "Eigen Matrix Result: " << std::endl;
std::cout << eigenMatrixResult << std::endl;
```

The above code will print out the following:

```
Eigen Matrix Result:
  0  24   1 -26
 30  52  14  12
  0  24   1 -26
 24  34  14  12
  0  18  -8 -37
```

@subsection vec_ops Vector Operations

Vector operations that IVSparse supports are norm, sum, and dot products with eigen vectors. Some notes are that because of the support for eigen vectors only in dot products its impossible to take the dot product of a IVSparse Vector with itself but this is something we intend to add in future work. Here are some examples:

```cpp
// norm
double norm = exampleVector.norm();
std::cout << "Norm: " << norm << std::endl;

// sum
double sum = exampleVector.sum();
std::cout << "Sum: " << sum << std::endl;

// dot product
double dotProduct = exampleVector.dot(eigenVector);
std::cout << "Dot Product: " << dotProduct << std::endl;
```

The above code will print out the following:

```
Norm: 6.40312
Sum: 11
Dot Product: 31
```

@subsection mat_ops Matrix Operations

There are a number of matrix operations supported for IVSparse compression formats. These are:

* Inner/Outer Sums
* Max/Min Col/Row Coefficients
* Trace
* Sum
* Norm
* Vector Length

These are supported for all compression formats and are fairly self explanitory.

```
Example Matrix:
4 0 3 0 2 
0 4 0 2 0 
4 0 3 0 1 
0 2 0 2 0 
3 0 4 0 3
```
*Outer/Inner Sums*

Outer sums returns a vector where each element is the sum of the corresponding outer dimension of the matrix. For column major matrices outer sum will return a vector of all sums for each column. Inner sums is the same but for the inner dimesion. This method is also very efficient for VCSC due to the value compression techniques.

```cpp
std::vector<double> outerSums = exampleMatrix.outerSums();

std::cout << "Outer Sums: " << std::endl;
for (auto sum : outerSums) {
    std::cout << sum << " ";
}
```

```
11 6 10 4 6
```

*Max/Min Col/Row Coefficients*

This is a set of four methods that return a vector of the minimum or maximum coefficents in either each column or row. These are used to find out information about the matrix such as the range of values across rows or columns. These are also very efficient for VCSC due to the value compression techniques.

```cpp
// Max column coefficients
std::vector<double> maxColCoeffs = exampleMatrix.maxColCoeffs();

// Min column coefficients
std::vector<double> minColCoeffs = exampleMatrix.minColCoeffs();

// range of column coefficients
std::vector<double> colRanges(maxColCoeffs.size());
for (int i = 0; i < maxColCoeffs.size(); i++) {
    colRanges[i] = maxColCoeffs[i] - minColCoeffs[i];
}

std::cout << "Column Ranges: " << std::endl;
for (auto range : colRanges) {
    std::cout << range << " ";
}
```

```
Column Ranges:
4 4 4 2 3
```

*Trace*

Trace returns the sum of the diagonal of the matrix. This is a very simple method that is supported for all compression formats.

```cpp
double trace = exampleMatrix.trace();

std::cout << "Trace: " << trace << std::endl;
```

```
Trace: 16
```

*Sum*

Sum returns the sum of all values in the matrix. This is a very simple method that is supported for all compression formats.

```cpp
double sum = exampleMatrix.sum();

std::cout << "Sum: " << sum << std::endl;
```

```
Sum: 37
```

*Norm*

Norm returns the Frobenius norm of the matrix. The Frobenius Norm is the sum of the squares of all the matrix coefficients. This is a very simple method that is supported for all compression formats.

```cpp
double norm = exampleMatrix.norm();

std::cout << "Norm: " << norm << std::endl;
```

```
Norm: 10.8167
```

*Vector Length*

Vector length returns the length of the indicated vector. This is calculated by taking the square root of the sum of squares of the vector coefficients. This is a very simple method that is supported for all compression formats.

```cpp
double vectorLength = exampleVector.vectorLength(0);

std::cout << "Vector Length: " << vectorLength << std::endl;
```

```
Vector Length: 6.40312
```
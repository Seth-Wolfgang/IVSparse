#include <chrono>
#include <iostream>

#include "IVSparse/SparseMatrix"
#include "misc/matrix_creator.cpp"

void getMat(Eigen::SparseMatrix<int>& myMatrix_e) {
  // declare an eigen sparse matrix of both types

  // col 0
  myMatrix_e.insert(0, 0) = 1;
  myMatrix_e.insert(2, 0) = 2;
  myMatrix_e.insert(3, 0) = 3;
  myMatrix_e.insert(5, 0) = 1;
  myMatrix_e.insert(6, 0) = 3;
  myMatrix_e.insert(7, 0) = 8;

  // col 1
  myMatrix_e.insert(3, 1) = 1;
  myMatrix_e.insert(4, 1) = 3;
  myMatrix_e.insert(5, 1) = 8;
  myMatrix_e.insert(6, 1) = 7;
  myMatrix_e.insert(8, 1) = 1;
  myMatrix_e.insert(9, 1) = 2;

  // col 2
  myMatrix_e.insert(0, 2) = 2;
  myMatrix_e.insert(2, 2) = 2;
  myMatrix_e.insert(5, 2) = 1;
  myMatrix_e.insert(7, 2) = 3;
  myMatrix_e.insert(9, 2) = 1;

  // col 3

  // col 4
  myMatrix_e.insert(0, 4) = 1;
  myMatrix_e.insert(3, 4) = 1;
  myMatrix_e.insert(4, 4) = 3;
  myMatrix_e.insert(6, 4) = 2;
  myMatrix_e.insert(7, 4) = 1;

  // col 5
  myMatrix_e.insert(0, 5) = 8;
  myMatrix_e.insert(2, 5) = 1;
  myMatrix_e.insert(3, 5) = 4;
  myMatrix_e.insert(5, 5) = 3;
  myMatrix_e.insert(7, 5) = 1;
  myMatrix_e.insert(8, 5) = 2;

  // col 6
  myMatrix_e.insert(3, 6) = 8;  // used to be a 6
  myMatrix_e.insert(5, 6) = 1;
  myMatrix_e.insert(7, 6) = 3;

  // col 7
  myMatrix_e.insert(2, 7) = 3;
  myMatrix_e.insert(4, 7) = 4;
  myMatrix_e.insert(5, 7) = 1;
  myMatrix_e.insert(8, 7) = 2;
  myMatrix_e.insert(9, 7) = 3;

  // col 8
  myMatrix_e.insert(0, 8) = 2;
  myMatrix_e.insert(2, 8) = 1;
  myMatrix_e.insert(3, 8) = 2;
  myMatrix_e.insert(5, 8) = 3;
  myMatrix_e.insert(7, 8) = 3;
  myMatrix_e.insert(9, 8) = 1;

  // col 9
  myMatrix_e.insert(3, 9) = 2;
  myMatrix_e.insert(4, 9) = 4;
  myMatrix_e.insert(7, 9) = 1;
  myMatrix_e.insert(8, 9) = 1;

  myMatrix_e.makeCompressed();
}

int main() {
  // get a sparse matrix
  Eigen::SparseMatrix<int> myMatrix_e(10, 10);

  getMat(myMatrix_e);

  std::cout << "myMatrix_e: " << std::endl;
  std::cout << myMatrix_e << std::endl;

  // make an IVSparse VCSC matrix
  IVSparse::SparseMatrix<int, int, 2, true> myMatrix(myMatrix_e);

  myMatrix.print();

  // transpose the matrix
  IVSparse::SparseMatrix<int, int, 2, true> myMatrixT = myMatrix.transpose();

  myMatrixT.print();

  return 0;
}
#include "IVSparse/SparseMatrix"
#include "misc/matrix_creator.cpp"

int main()
{

    // For ease of use, create an Eigen::SparseMatrix
    Eigen::SparseMatrix<int> eigenSparseMatrix(5, 5);
    eigenSparseMatrix = generateMatrix<int>(5, 5, 2, 0, 4);
    eigenSparseMatrix.makeCompressed();

    Eigen::MatrixXi mat(5, 4);
    mat <<  0, 6, 4, 1,
            3, 9, 0, 0,
            0, 0, -5, -10,
            9, 8, 7, 6,
            0, 0, 0, 0;
    Eigen::SparseMatrix<int> CSC_Mat = mat.sparseView();

    // std::cout << mat << std::endl;

    // Create a IVSparse::SparseMatrix from the Eigen::SparseMatrix
    IVSparse::SparseMatrix<int> IVCSC_SparseMatrix(eigenSparseMatrix);

    // Print the IVSparse::SparseMatrix
    // std::cout << "Construction: " << std::endl;
    // std::cout << IVCSC_SparseMatrix << std::endl;

    /**
     * Scalar multiplication
     *
     * Scalar multiplication iterates through the whole matrix and scalese each value by the scalar.
     * This method is O(n^2) for IVSparse 3, but for IVSparse 2 it will only iterate through the unique values stored in
     * performance arrays if the setting is active.
     *
     * In place operations are also supported.
     *
     */
    //IVCSC_SparseMatrix = IVCSC_SparseMatrix * 2;

    // Print the IVSparse::SparseMatrix
    // std::cout << "\nScalar Multiplication" << std::endl;
    // std::cout << IVCSC_SparseMatrix << std::endl;

    /**
     * Vector multiplication
     *
     * Vector multiplication iterates through the whole matrix and multiplies each value by the corresponding value in the vector.
     * This is done in O(n^2) and is slower than the Eigen::SparseMatrix implementation.
     *
     * The product does produce an Eigen::Vector as opposed to a IVSparse::SparseMatrix.
     *
     */
    Eigen::VectorXi eigenVector(5);
    eigenVector << 1, 2, 3, 4, 5;

    std::cout << eigenVector << std::endl;

    IVSparse::SparseMatrix<int>::Vector vec(IVCSC_SparseMatrix, 0);

    vec.print();

    double dotP = vec.dot(eigenVector);

    std::cout << "Dot Product: " << dotP << std::endl;

    std::cout << "Norm: " << vec.norm() << std::endl;

    std::cout << "Sum: " << vec.sum() << std::endl;

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    // Eigen::VectorXi eigenResult = IVCSC_SparseMatrix * eigenVector;

    // // Print the Eigen::VectorXd
    // std::cout << "\nSpM * V Multiplication" << std::endl;
    // std::cout << eigenResult << std::endl;

    /**
     * Matrix multiplication
     *
     * Our matrix mutliplication method produces an Eigen::Matrix as the product of a IVSparse::SparseMatrix and an Eigen::Matrix.
     * As of right now, this only works for Sparse * Dense operations, so it is not currently possible to multiply by an Eigen::SparseMatrix.
     *
     * This algorithm is O(n^3).
     *
     */
    // Eigen::MatrixXi eigenMatrixResult = IVCSC_SparseMatrix * mat; // Multiply the IVSparse::SparseMatrix by the Eigen::Matrix

    /**
     * Matrix Transpose
     *
     * The matrix transpose method produces a transposed version of the IVSparse::SparseMatrix. Note: This operation can be quite costly
     * due to how IVSparse::SparseMatrix is stored.
     *
     * This algorithm is in O(n^2)
     */
    // IVSparse::SparseMatrix<int> IVCSC_SparseMatrix_T = IVCSC_SparseMatrix.transpose();

    // // Print the Eigen::MatrixXd
    // std::cout << "\nSpM * M Multiplication" << std::endl;
    // std::cout << eigenMatrixResult << std::endl;

    /**
     * Outer Sums
     *
     * The column sum method adds up each elemnt in each outer index and returns an array containing the sums.
     * This method benefits greatly from the IVSparse 2 performance arrays and is recommnded to use them for this method
     *
     * This algorithm is in O(n^2)
     */

    std::vector<int> columnSums = IVCSC_SparseMatrix.outerSum();

    // Print the column sums
    std::cout << "\nOuter Sums" << std::endl;
    for (int i = 0; i < IVCSC_SparseMatrix.cols(); i++)
    {
        std::cout << columnSums[i] << std::endl;
    }

    /**
     * Row Sums
     *
     * The row sum method adds up each elemnt in each inner index and returns an array containing the sums.
     * This method does NOT benefit from IVSparse 2 performance arrays, but they won't hinder it either.
     * This method must iterate through all data stored in the IVSparse::SparseMatrix.
     *
     * This algorithm is in O(n^2)
     */

    std::vector<int> rowSums = IVCSC_SparseMatrix.innerSum();

    // Print the row sums
    std::cout << "\nInner Sums" << std::endl;
    for (int i = 0; i < IVCSC_SparseMatrix.rows(); i++)
    {
        std::cout << rowSums[i] << std::endl;
    }

    /**
     * Max outer coefficients
     *
     * The max outer coefficients method finds the maximum value in each outer index and returns an array containing the maximums.
     * This method benefits greatly from the IVSparse 2 performance arrays and is recommnded to use them for this method
     *
     * This algorithm is in O(n^2)
     */

    std::vector<int> maxOuterCoefficients = IVCSC_SparseMatrix.maxColCoeff();
    std::vector<int> minOuterCoefficientstest = IVCSC_SparseMatrix.minColCoeff();

    std::vector<int> ranges(IVCSC_SparseMatrix.cols());
    for (int i = 0; i < IVCSC_SparseMatrix.cols(); i++)
    {
        ranges[i] = maxOuterCoefficients[i] - minOuterCoefficientstest[i];
    }

    std::cout << "Ranges: " << std::endl;
    for (int i = 0; i < IVCSC_SparseMatrix.cols(); i++)
    {
        std::cout << ranges[i] << std::endl;
    }

    // Print the max outer coefficients
    std::cout << "\nMax Outer Coefficients" << std::endl;
    for (int i = 0; i < IVCSC_SparseMatrix.cols(); i++)
    {
        std::cout << maxOuterCoefficients[i] << std::endl;
    }

    /**
     * Max inner coefficients
     *
     * The max inner coefficients method finds the maximum value in each inner index and returns an array containing the maximums.
     * This method does NOT benefit from IVSparse 2 performance arrays, but they won't hinder it either.
     * This method must iterate through all data stored in the IVSparse::SparseMatrix.
     *
     * This algorithm is in O(n^2)
     */

    std::vector<int> maxInnerCoefficients = IVCSC_SparseMatrix.maxRowCoeff();

    // Print the max inner coefficients
    std::cout << "\nMax Inner Coefficients" << std::endl;
    for (int i = 0; i < IVCSC_SparseMatrix.rows(); i++)
    {
        std::cout << maxInnerCoefficients[i] << std::endl;
    }

    /**
     * Min outer coefficients
     *
     * The min outer coefficients method finds the minimum value in each outer index and returns an array containing the minimums.
     * This method benefits greatly from the IVSparse 2 performance arrays and is recommnded to use them for this method
     *
     * This algorithm is in O(n^2)
     */

    std::vector<int> minOuterCoefficients = IVCSC_SparseMatrix.minColCoeff();

    // Print the min outer coefficients
    std::cout << "\nMin Outer Coefficients" << std::endl;
    for (int i = 0; i < IVCSC_SparseMatrix.cols(); i++)
    {
        std::cout << minOuterCoefficients[i] << std::endl;
    }

    /**
     * Min inner coefficients
     *
     * The min inner coefficients method finds the minimum value in each inner index and returns an array containing the minimums.
     * This method does NOT benefit from IVSparse 2 performance arrays, but they won't hinder it either.
     * This method must iterate through all data stored in the IVSparse::SparseMatrix.
     *
     * This algorithm is in O(n^2)
     */

    std::vector<int> minInnerCoefficients = IVCSC_SparseMatrix.minRowCoeff();

    // Print the min inner coefficients
    std::cout << "\nMin Inner Coefficients" << std::endl;
    for (int i = 0; i < IVCSC_SparseMatrix.rows(); i++)
    {
        std::cout << minInnerCoefficients[i] << std::endl;
    }

    /**
     * Trace of a matrix
     *
     * The trace of a matrix is the sum of the diagonal elements of the matrix.
     * This method does NOT benefit from IVSparse 2 performance arrays, but they won't hinder it either.
     *
     * This algorithm is in O(n^2)
     */

    int trace = IVCSC_SparseMatrix.trace();

    // Print the trace
    std::cout << "\nTrace: " << trace << std::endl;

    /**
     * Sum of Matrix Coefficients
     *
     * The sum of matrix coefficients is the sum of all elements in the matrix.
     * This method benefits greatly from IVSparse 2 performance arrays and is highly
     * recommnded to use them for this method
     *
     * This algorithm is in O(n^2)
     */

    int sum = IVCSC_SparseMatrix.sum();

    // Print the sum
    std::cout << "Sum of Coefficients: " << sum << std::endl;

    /**
     * Frobenius Norm
     *
     * The Frobenius norm is the square root of the sum of the squares of the matrix coefficients.
     * This method benefits greatly from IVSparse 2 performance arrays and is highly
     * recommnded to use them for this method
     *
     * This algorithm is in O(n^2)
     */

    double frobeniusNorm = IVCSC_SparseMatrix.norm();

    // Print the Frobenius norm
    std::cout << "Frobenius Norm: " << frobeniusNorm << std::endl;

    /**
     * Vector Length
     *
     * The vector length is the square root of the sum of the squares of the vector coefficients.
     * This method benefits greatly from IVSparse 2 performance arrays and is highly
     * recommnded to use them for this method
     *
     * This algorithm is in O(n)
     */

    double vectorLength = IVCSC_SparseMatrix.vectorLength(0); // This is the length of the first vector

    // Print the vector length
    std::cout << "Length of first vector: " << vectorLength << std::endl;

    return 0;
}
#include "../IVSparse/SparseMatrix"
#include "../misc/matrix_creator.cpp"

int main()
{

    /**********************************************************************************************
     *                                                                                            *
     *                                                                                            *
     *                                          MATRICES                                          *
     *                                                                                            *
     *                                                                                            *
     *                                                                                            *
     *                                                                                            *
     * *******************************************************************************************/

    // Create an Eigen matrix so we can convert to IVSparse
    Eigen::SparseMatrix<double> eigenMatrix(4, 20);
    eigenMatrix = generateMatrix<double>(4, 20, 1, 0, 3);        // A function to generate a random matrix
    IVSparse::SparseMatrix<double> csfSparseMatrix(eigenMatrix); // Create a IVSparse::SparseMatrix from the Eigen::MatrixXd

    /**
     * The IVSparse::SparseMatrix class has an iterator that can be used to traverse through the data.
     * The iterator is a nested class within the IVSparse::SparseMatrix class, so it is important to specify 'typename' before the iterator.
     */

    /**
     * The iterator works by moving through the data in the IVSparse::SparseMatrix.
     * To iterate through the data, we use the "++" operator.
     *
     * The example below shows the iterator being set to the beginning of column 0.

     * Note: This is very similar to the Eigen::SparseMatrix innerIterator.
     *
     */

    for (typename IVSparse::SparseMatrix<double>::InnerIterator it(csfSparseMatrix, 0); it; ++it)
    {
        std::cout << it.value() << " ";
    }
    std::cout << std::endl
              << std::endl;

    /**
     * As shown above, we can get the value from the iterator by using the "value()" method.
     * We can also use the * operator to get the value.
     * Note: The values are not in a particular order due to how the matrix is stored.
     */

    for (typename IVSparse::SparseMatrix<double>::InnerIterator it(csfSparseMatrix, 0); it; ++it)
    {
        std::cout << *it << " ";
    }
    std::cout << std::endl
              << std::endl;

    /**
     * We can also get the row and column of the iterator by using the "row()" and "col()" methods.
     */

    for (typename IVSparse::SparseMatrix<double>::InnerIterator it(csfSparseMatrix, 0); it; ++it)
    {
        std::cout << it.row() << " " << it.col() << std::endl;
    }
    std::cout << std::endl;

    /**
     * To Traverse through the whole matrix, we simply add another for loop.
     */

    for (uint32_t i = 0; i < csfSparseMatrix.outerSize(); ++i)
    {
        for (typename IVSparse::SparseMatrix<double>::InnerIterator it(csfSparseMatrix, i); it; ++it)
        {
            std::cout << it.value() << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    /**
     * It is also possible to change values within the matrix using the iterator. However, this is dangerous do to
     * how data is stored in a IVSparse matrix. **If you change any value then that value will be changed for ALL occurrences of the value changed.**
     *
     * Ex.
     * if our vector looks like this -> Value: 3 indices: [1, 2, 3, 4, 5, 6, 7, 8, 9] and we change the value to 17
     * Then our vector will look like -> Value: 17 indices: [1, 2, 3, 4, 5, 6, 7, 8, 9].
     *
     * Notice they are very similar, but the value is the only part that changes, and it changes for all occurrences of the value.
     */

    // Here we will create a random Eigen::SparseMatrix with 20 columns, 4 rows, and a max value of 5
    // Eigen::SparseMatrix<int> myEigenMatrix(20, 4);
    // eigenMatrix = generateMatrix<int>(20, 4, 1, 0, 5);
    IVSparse::SparseMatrix<double, int, 3> csfMatrix(eigenMatrix);

    std::cout << csfSparseMatrix << std::endl;

    // Now we will change the first value in the matrix to a 17
    typename IVSparse::SparseMatrix<double>::InnerIterator myIter(csfSparseMatrix, 0);
    myIter.coeff(17);

    std::cout << csfSparseMatrix << std::endl;

    /**
     * Notice that several values are now changed
     */

    /**********************************************************************************************
     *                                                                                            *
     *                                                                                            *
     *                                          VECTORS                                           *
     *                                                                                            *
     *                                                                                            *
     *                                                                                            *
     * *******************************************************************************************/

    /**
     *
     * The InnerIterator is also capable of iterating across IVSparse::Vector
     *
     */

    typename IVSparse::SparseMatrix<double>::Vector myVec = csfSparseMatrix.getVector(0);
    /**
     * To iterate over the vector, you can simply
     */

    for (typename IVSparse::SparseMatrix<double>::InnerIterator it(myVec); it; ++it)
    {
        std::cout << it.value() << " ";
    }
}

#include <iostream>
#include "IVSparse/SparseMatrix"
#include "misc/matrix_creator.cpp"
#include <chrono>

#define DATA_TYPE int
#define INDEX_TYPE int

template <typename T, typename indexT>
void sizeTest(int iterations);

template <typename T, typename indexT, int compressionLevel>
void iteratorTest();
void getMat(Eigen::SparseMatrix<int>& myMatrix_e);

template <typename T>
void generateAllUniqueElements(Eigen::SparseMatrix<T>& eigen);

template <typename T>
void generateAllRedundantElements(Eigen::SparseMatrix<T>& eigen);

template <typename T>
bool compareMatrices(Eigen::Matrix<T, -1, -1> mat1, Eigen::Matrix<T, -1, -1> mat2, Eigen::Matrix<T, -1, -1> mat3);

template <typename T>
std::vector<std::tuple<int, int, int>> generateCOO(int rows, int cols, int max);

//  For my convenience
//  clear; rm a.out; g++ test.cpp; ./a.out

int main() {
    
    int rows = 10000;
    int cols = 100;
    int sparsity = 5;
    uint64_t seed = 522;
    int maxVal = 10;
    const bool isColMajor = true;

    Eigen::MatrixXi testDense = Eigen::MatrixXi::Random(rows-1, cols);
    Eigen::MatrixXi testDense2 = Eigen::MatrixXi::Random(cols, rows + 3);
    Eigen::SparseMatrix<DATA_TYPE> testEigenDense = testDense.sparseView();
    IVSparse::SparseMatrix<DATA_TYPE, INDEX_TYPE, 3, isColMajor> csf3(testEigenDense);
    IVSparse::SparseMatrix<DATA_TYPE, INDEX_TYPE, 2, isColMajor> csf2(testEigenDense);
    Eigen::VectorXi eigenVec = Eigen::VectorXi::Random(cols);

    // std::cout << (csf3 * testDense2) << std::endl << std::endl << std::endl;
    // std::cout << (csf2 * testDense2) << std::endl << std::endl << std::endl;


    // std::cout << (testEigenDense * testDense2) << std::endl;


    std::cout << "IVCSC:   " << (csf3 * testDense2).sum() << std::endl;
    std::cout << "VCSC:    " << (csf2 * testDense2).sum() << std::endl;
    std::cout << "them:  " << (testEigenDense * testDense2).sum() << std::endl;

    std::vector<uint64_t> csf3Times;
    std::vector<uint64_t> csf2Times;
    std::vector<uint64_t> eigenTimes;
    // return 1;

    printf("%10s %10s %10s\n", "VCSC", "IVCSC", "Eigen");
    
    for (int i = 0; i < 100; i++) {
        srand(time(NULL));

        // rows = rand() % 100 + 1;
        // cols = rand() % 100 + 1;
        Eigen::SparseMatrix<int> original = generateMatrix<int>(rows, cols, sparsity, seed, maxVal);

        // std::cout << "iteration: " << i << std::endl;
        // std::cout << "Rows: " << rows << " Cols: " << cols << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;

        std::stringstream os1;
        std::stringstream os2;
        std::stringstream os3;

        Eigen::MatrixXi dense = Eigen::MatrixXi::Random(rows, cols);
        // Eigen::MatrixXi dense2 = Eigen::MatrixXi::Random(cols, rows);
        dense.fill(1);
        // std::cout << "Dense: " << dense << std::endl;
        Eigen::SparseMatrix<int> eigenDense = dense.sparseView();
        IVSparse::SparseMatrix<int, INDEX_TYPE, 3, isColMajor> csf3(eigenDense);
        IVSparse::SparseMatrix<int, INDEX_TYPE, 2, isColMajor> csf2(eigenDense);

        Eigen::VectorXi eigenVec = Eigen::VectorXi::Random(cols);

        // std::cout << "rows in vec:" << eigenVec.rows() << std::endl;
        // std::cout << "Rows and cols in dense: " << dense.rows() << " " << dense.cols() << std::endl;
        // std::cout << "Rows and cols in csf3: " << csf3.rows() << " " << csf3.cols() << std::endl;

        start = std::chrono::system_clock::now();
        Eigen::MatrixXi filler1 = csf3 * eigenVec;
        end = std::chrono::system_clock::now();
        uint64_t csf3Time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::system_clock::now();
        Eigen::MatrixXi filler2 = csf2 * eigenVec;
        end = std::chrono::system_clock::now();
        uint64_t csf2Time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::system_clock::now();
        Eigen::MatrixXi filler3 = eigenDense * eigenVec;
        end = std::chrono::system_clock::now();
        uint64_t eigenTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();


        // std::cout << "filler1" << std::endl;
        // std::cout << filler1 << std::endl;

        // std::cout << "filler2" << std::endl;
        // std::cout << filler2 << std::endl;

        // std::cout << "filler3" << std::endl;
        // std::cout << filler3 << std::endl;

        // std::cout << "Our sum: " << ourSum << " Their sum: " << theirSum << std::endl;
        // std::cout << "Sum: " << csf3Sum << " " << csf2Sum << " " << eigenSum << std::endl;
        assert(compareMatrices(filler1, filler2, filler3));
        // std::cout << i << ": Works!" << std::endl;
        // assert(sum1 == sum2);
        // assert(sum2 == sum3);
        printf("%10lu %10lu %10lu\n", csf2Time, csf3Time, eigenTime);
        eigenTimes.push_back(eigenTime);
        csf2Times.push_back(csf2Time);
        csf3Times.push_back(csf3Time);
    }

    uint64_t avgCSF2Time = 0;
    uint64_t avgEigenTime = 0;
    uint64_t avgCSF3Time = 0;

    for (uint32_t i = 0; i < csf2Times.size(); i++) {
        avgCSF2Time += csf2Times[i];
        avgEigenTime += eigenTimes[i];
        avgCSF3Time += csf3Times[i];
    }

    avgCSF2Time /= csf2Times.size();
    avgEigenTime /= eigenTimes.size();
    avgCSF3Time /= csf3Times.size();

    std::cout << "VCSC:  " << avgCSF2Time << std::endl;
    std::cout << "IVCSC: " << avgCSF3Time << std::endl;
    std::cout << "Eigen: " << avgEigenTime << std::endl;

    std::cout << "Eigen takes " << (double)avgEigenTime / avgCSF2Time << " times as long as VCSC" << std::endl;
    std::cout << "Eigen takes " << (double)avgEigenTime / avgCSF3Time << " times as long as IVCSC" << std::endl;

    return 0;
}

template <typename T>
bool compareMatrices(Eigen::Matrix<T, -1, -1> mat1, Eigen::Matrix<T, -1, -1> mat2, Eigen::Matrix<T, -1, -1> mat3) {

    if (mat1.cols() != mat2.cols() || mat2.cols() != mat3.cols()) {
        std::cout << "mat1: " << mat1.cols() << " mat2: " << mat2.cols() << " mat3: " << mat3.cols() << std::endl;
        return false;
    }
    if (mat1.rows() != mat2.rows() || mat2.rows() != mat3.rows()) {
        std::cout << "mat1: " << mat1.rows() << " mat2: " << mat2.rows() << " mat3: " << mat3.rows() << std::endl;
        return false;
    }
    if (mat1.sum() != mat2.sum() || mat2.sum() != mat3.sum()) {
        std::cout << "mat1: " << mat1.sum() << " mat2: " << mat2.sum() << " mat3: " << mat3.sum() << std::endl;
        return false;
    }

    for (int i = 0; i < mat3.rows(); i++) {
        for (int j = 0; j < mat3.cols(); j++) {
            if (mat1(i, j) != mat2(i, j) || mat2(i, j) != mat3(i, j)) {
                std::cout << "mat1: " << mat1(i, j) << " mat2: " << mat2(i, j) << " mat3: " << mat3(i, j) << std::endl;
                return false;
            }
        }
    }
    return true;

}

template <typename T, typename indexT>
void sizeTest(int iterations) {
    int rows = 100;
    int cols = 100;
    int sparsity = 9;
    uint64_t seed = 1;
    int maxVal = 1000;

    std::cout << "Rows: " << rows << " \nCols: " << cols << " \nSparsity: " << sparsity << " \nSeed: " << seed << " \nMaxVal " << maxVal << std::endl;

    std::vector<uint64_t> csf2Sizes;
    std::vector<uint64_t> csfSizes;

    #pragma omp parallel for num_threads(15)
    for (int i = 0; i < iterations; i++) {
        // create an eigen sparse matrix
        Eigen::SparseMatrix<T> eigen(rows, cols);
        // getMat(eigen);
        eigen = generateMatrix<T>(rows, cols, sparsity, rand(), maxVal);
        // std::cout << eigen << std::endl;

        // create a IVSparse sparse matrix
        IVSparse::SparseMatrix<T, indexT, 3> csf(eigen);
        IVSparse::SparseMatrix<T, indexT, 2> csf2(eigen);

        csfSizes.push_back(csf.compressionSize());
        csf2Sizes.push_back(csf2.compressionSize());
    }

    uint64_t avgCSF2Size = 0;
    uint64_t avgCSFSize = 0;
    for (int i = 0; i < csf2Sizes.size(); i++) {
        avgCSF2Size += csf2Sizes[i];
        avgCSFSize += csfSizes[i];
    }
    avgCSF2Size /= csf2Sizes.size();
    avgCSFSize /= csfSizes.size();

    std::cout << "IVSparse: " << avgCSFSize << std::endl;
    std::cout << "VCSC: " << avgCSF2Size << std::endl;
    // uint64_t eigenSize = eigen.nonZeros() * sizeof(double) + eigen.nonZeros() * sizeof(uint32_t) + (eigen.outerSize() + 1) * sizeof(uint32_t);
    // std::cout << "eigen size: " << eigenSize << std::endl;
}

template <typename T, typename indexT, int compressionLevel>
void iteratorTest() {

    int numRows = 10000; // rand() % 1000 + 10;
    int numCols = 10000; // rand() % 1000 + 10;
    int sparsity = 1;    // rand() % 50 + 1;
    uint64_t seed = 1;   // rand();

    // Initialize the random matrix
    Eigen::SparseMatrix<T> eigen(numRows, numCols);
    eigen.reserve(Eigen::VectorXi::Constant(numCols, numRows));
    eigen = generateMatrix<T>(numRows, numCols, sparsity, rand(), 1);
    eigen.makeCompressed();

    // Create random matrix and vector to multiply with
    //  Eigen::Matrix<T, -1, -1> randMatrix = Eigen::Matrix<T, -1, -1>::Random(numCols, numRows);
    //  Eigen::Matrix<T, -1, 1> randVector = Eigen::Matrix<T, -1, 1>::Random(numCols);

    // Eigen::VectorXd randVector = Eigen::VectorXd::Random(numCols);

    // Create IVSparse matrix and an eigen dense matrix
    IVSparse::SparseMatrix<T, indexT, compressionLevel> csfMatrix(eigen);

    // Create a dense matrix to store the result of the multiplication
    std::chrono::time_point<std::chrono::system_clock> start, end;
    // Eigen::Matrix<T, -1, -1>  csfDenseMatrix;
    // Eigen::Matrix<T, -1, -1> eigenDenseMatrix;

    // Vectors to store times for averages
    std::vector<uint64_t> timesForNew;
    std::vector<uint64_t> timesForOld;
    uint64_t ours = 0;
    uint64_t old = 0;
    for (int i = 0; i < 1; i++) {

        // Measure time for IVSparse matrix
        T sum = 0;
        start = std::chrono::system_clock::now();

        for (int i = 0; i < csfMatrix.outerSize(); ++i) {
            for (typename IVSparse::SparseMatrix<T, indexT, compressionLevel>::InnerIterator it(csfMatrix, i); it; ++it) {
                sum += it.value();
            }
        }

        end = std::chrono::system_clock::now();
        timesForNew.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        ours = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // std::cout << "IVSparse:\n " << csfDenseMatrix << std::endl;

        // Measure time for Eigen matrix
        T sum2 = 0;
        start = std::chrono::system_clock::now();
        for (int i = 0; i < eigen.outerSize(); ++i) {
            for (typename Eigen::SparseMatrix<T>::InnerIterator it(eigen, i); it; ++it) {
                sum2 += it.value();
            }
        }
        end = std::chrono::system_clock::now();
        timesForOld.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        old = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        assert(sum2 == sum);
        // std::cout << "(IVSparse): " << ours << "(Eigen): " << old << std::endl;

        // std::cout << "Eigen:\n " << eigenDenseMatrix << std::endl;
    }
    // take average of timesforNew and timesForOld
    uint64_t duration = 0;
    uint64_t duration2 = 0;
    for (int i = 0; i < timesForNew.size(); i++) {
        duration += timesForNew[i];
        duration2 += timesForOld[i];
    }
    duration /= timesForNew.size();
    duration2 /= timesForOld.size();

    std::cout << "Version 1 (IVSparse): " << duration << " version 2 (Eigen): " << duration2 << std::endl;

    // Eigen::MatrixXd controlMatrix = eigen * randMatrix;

    // T sum_e = denseMatrix.sum();
    // T sumCSF = csfDenseMatrix.sum();
    // T sumEigen = eigenDenseMatrix.sum();

    // std::cout << "Eigen: " << sumEigen << " IVSparse: " << sumCSF << std::endl;

    // if (sumCSF == 0 || sumEigen == 0 || sumCSF != sumEigen) {
    //     std::cout << "Rows: " << numRows << " Cols: " << numCols << " Sparsity: " << sparsity << " Seed: " << seed << std::endl;
    //     std::cout << "sum_csf: " << sumCSF << " Eigen: " << sumEigen << std::endl;
    //     assert(sumCSF == sumEigen);
    // }
}

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
    myMatrix_e.insert(3, 6) = 6;
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

template <typename T>
void generateAllUniqueElements(Eigen::SparseMatrix<T>& eigen) {
    T count = 1;
    std::cout << "Cols: " << eigen.cols() << " Rows: " << eigen.rows() << std::endl;
    std::cout << "Total values: " << eigen.cols() * eigen.rows() << std::endl;
    for (int i = 0; i < eigen.cols(); i++) {
        for (int j = 0; j < eigen.rows(); j++) {
            // std::cout << "Inserting: " << count << std::endl;
            eigen.insert(j, i) = (T)(count++);
        }
    }
}

template <typename T>
void generateAllRedundantElements(Eigen::SparseMatrix<T>& eigen) {
    T count = 1;
    std::cout << "Cols: " << eigen.cols() << " Rows: " << eigen.rows() << std::endl;
    std::cout << "Total values: " << eigen.cols() * eigen.rows() << std::endl;
    for (int i = 0; i < eigen.cols(); i++) {
        for (int j = 0; j < eigen.rows(); j++) {
            // std::cout << "Inserting: " << count << std::endl;
            eigen.insert(j, i) = count;
        }
    }
}

template <typename T>
std::vector<std::tuple<int, int, int>> generateCOO(int rows, int cols, int max) {
    std::vector<std::tuple<int, int, int>> coo;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            coo.push_back(std::make_tuple(i, j, rand() % max));
        }
    }
    return coo;
}
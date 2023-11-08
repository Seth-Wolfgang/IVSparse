/**
 * @brief A simple class to store and record basic statistical data of benchmarking results
 * @author Seth Wolfgang
 * @date 2023-03-22
 */

#include <stdio.h>
#include <vector>
#include <unistd.h>
#include <stdlib.h>
#include <inttypes.h>

class BenchAnalysis {
    private:
    typedef struct matrixInfo {
        double matrixNum;
        double matrixRows;
        double matrixCols;
        double matrixNonzeros;
        double matrixRedundancy;
        double matrixDensity;
    } matrix;

    matrix* myMatrix;

    // Constructor
    std::vector<uint64_t> EigenConstructorTimes;
    std::vector<uint64_t> CSF1ConstructorTimes;
    std::vector<uint64_t> CSF2ConstructorTimes;
    std::vector<uint64_t> CSF3ConstructorTimes;
    std::vector<uint64_t> ArmadilloConstructorTimes;

    // Iterator
    std::vector<uint64_t> EigenInnerIteratorTimes;
    std::vector<uint64_t> CSF1InnerIteratorTimes;
    std::vector<uint64_t> CSF2InnerIteratorTimes;
    std::vector<uint64_t> CSF3InnerIteratorTimes;
    std::vector<uint64_t> ArmadilloInnerIteratorTimes;

    // Scalar Multiplication
    std::vector<uint64_t> EigenScalarMultiplicationTimes;
    std::vector<uint64_t> CSF1ScalarMultiplicationTimes;
    std::vector<uint64_t> CSF2ScalarMultiplicationTimes;
    std::vector<uint64_t> CSF3ScalarMultiplicationTimes;
    std::vector<uint64_t> ArmadilloScalarMultiplicationTimes;

    // Vector Multiplication
    std::vector<uint64_t> EigenVectorMultiplicationTimes;
    std::vector<uint64_t> CSF1VectorMultiplicationTimes;
    std::vector<uint64_t> CSF2VectorMultiplicationTimes;
    std::vector<uint64_t> CSF3VectorMultiplicationTimes;
    std::vector<uint64_t> ArmadilloVectorMultiplicationTimes;

    // memory usage (in bytes)
    std::vector<uint64_t> EigenMemoryUsage;
    std::vector<uint64_t> CSF1MemoryUsage;
    std::vector<uint64_t> CSF2MemoryUsage;
    std::vector<uint64_t> CSF3MemoryUsage;
    std::vector<uint64_t> ArmadilloMemoryUsage;

    // Transpose benchmark
    std::vector<uint64_t> EigenTransposeTimes;
    std::vector<uint64_t> CSF1TransposeTimes;
    std::vector<uint64_t> CSF2TransposeTimes;
    std::vector<uint64_t> CSF3TransposeTimes;
    std::vector<uint64_t> ArmadilloTransposeTimes;

    // Matrix Multiplication
    std::vector<uint64_t> EigenMatrixMultiplicationTimes;
    std::vector<uint64_t> CSF1MatrixMultiplicationTimes;
    std::vector<uint64_t> CSF2MatrixMultiplicationTimes;
    std::vector<uint64_t> CSF3MatrixMultiplicationTimes;
    std::vector<uint64_t> ArmadilloMatrixMultiplicationTimes;

    // Outer Sum
    std::vector<uint64_t> EigenOuterSumTimes;
    std::vector<uint64_t> CSF1OuterSumTimes;
    std::vector<uint64_t> CSF2OuterSumTimes;
    std::vector<uint64_t> CSF3OuterSumTimes;
    std::vector<uint64_t> ArmadilloOuterSumTimes;

    // Sum
    std::vector<uint64_t> EigenSumTimes;
    std::vector<uint64_t> CSF1SumTimes;
    std::vector<uint64_t> CSF2SumTimes;
    std::vector<uint64_t> CSF3SumTimes;
    std::vector<uint64_t> ArmadilloSumTimes;

    // Norm
    std::vector<uint64_t> EigenNormTimes;
    std::vector<uint64_t> CSF1NormTimes;
    std::vector<uint64_t> CSF2NormTimes;
    std::vector<uint64_t> CSF3NormTimes;
    std::vector<uint64_t> ArmadilloNormTimes;

    public:
    // Constructor
    BenchAnalysis(std::vector<double> matrixData) {
        myMatrix = (matrix*)malloc(sizeof(matrix));
        // matrixData -> ID, rows, cols, nonzeos, redundancy, matrix density
        //  Assigning the matrix data to the struct
        myMatrix->matrixNum = matrixData[0];
        myMatrix->matrixRows = matrixData[1];
        myMatrix->matrixCols = matrixData[2];
        myMatrix->matrixNonzeros = matrixData[3];
        myMatrix->matrixRedundancy = matrixData[4];
        myMatrix->matrixDensity = matrixData[5];

        // Checking to see if the files we're writing to exist in the directory
        // if not, we write the headers at the top of the file.
        FILE* file;
        if (access("timeData.csv", F_OK) != 0) {
            std::cout << "Creating timeData.csv" << std::endl;
            file = fopen("timeData.csv", "a");
            fprintf(file, "%s\n", "Matrix Number,Matrix Rows,Matrix Cols,Matrix Nonzeros,Matrix Redundancy,Matrix Density,Avg Eigen Constructor Time,Avg CSC Constructor Time,Avg CSF2 Constructor Time,Avg CSF3 Constructor Time,Avg Armadillo Constructor Time,Avg Eigen InnerIterator Time,Avg CSC InnerIterator Time,Avg CSF2 InnerIterator Time,Avg CSF3 InnerIterator Time,Avg Armadillo InnerIterator Time,Avg Eigen Scalar Multiplication Time,Avg CSC Scalar Multiplication Time,Avg CSF2 Scalar Multiplication Time,Avg CSF3 Scalar Multiplication Time,Avg Armadillo Scalar Multiplication Time,Avg Eigen Vector Multiplication Time,Avg CSC Vector Multiplication Time,Avg CSF2 Vector Multiplication Time,Avg CSF3 Vector Multiplication Time,Avg Armadillo Vector Multiplication Time,Avg Eigen Memory Usage,Avg CSC Memory Usage,Avg CSF2 Memory Usage,Avg CSF3 Memory Usage,Avg Armadillo Memory Usage,Avg Eigen Transpose Time,Avg CSCTranspose Time,Avg CSF2 Transpose Time,Avg CSF3 Transpose Time,Avg Armadillo Transpose Time,Avg Eigen Matrix Multiplication Time,Avg CSCMatrix Multiplication Time,Avg CSF2 Matrix Multiplication Time,Avg CSF3 Matrix Multiplication Time,Avg Armadillo Matrix Multiplication Time,Avg Eigen Outer Sum Time,Avg CSC Outer Sum Time,Avg CSF2 Outer Sum Time,Avg CSF3 Outer Sum Time,Avg Armadillo Outer Sum Time,Avg Eigen Sum Time,Avg CSC Sum Time,Avg CSF2 Sum Time,Avg CSF3 Sum Time,Avg Armadillo Sum Time,Avg Eigen Norm Time,Avg CSC Norm Time,Avg CSF2 Norm Time,Avg CSF3 Norm Time,Avg Armadillo Norm Time");
            fclose(file);
        }

        if (access("rawTimeData.csv", F_OK) != 0) {
            std::cout << "Creating rawTimeData.csv" << std::endl;
            file = fopen("rawTimeData.csv", "a");
            fprintf(file, "%s\n", "Matrix Number,Matrix Rows,Matrix Cols,Matrix Nonzeros,Matrix Redundancy,Matrix Density,Eigen Constructor Time,CSC Constructor Time,CSF2 Constructor Time,CSF3 Constructor Time,Armadillo Constructor Time,Eigen InnerIterator Time,CSC InnerIterator Time,CSF2 InnerIterator Time,CSF3 InnerIterator Time,Armadillo InnerIterator Time,Eigen Scalar Multiplication Time,CSC Scalar Multiplication Time,CSF2 Scalar Multiplication Time,CSF3 Scalar Multiplication Time,Armadillo Scalar Multiplication Time,Eigen Vector Multiplication Time,CSC Vector Multiplication Time,CSF2 Vector Multiplication Time,CSF3 Vector Multiplication Time,Armadillo Vector Multiplication Time,Eigen Memory Usage,CSC Memory Usage,CSF2 Memory Usage,CSF3 Memory Usage,Armadillo Memory Usage,Eigen Transpose Time,CSC Transpose Time,CSF2 Transpose Time,CSF3 Transpose Time,Armadillo Transpose Time,Eigen Matrix Multiplication Time,CSC Matrix Multiplication Time,CSF2 Matrix Multiplication Time,CSF3 Matrix Multiplication Time,Armadillo Matrix Multiplication Time,Eigen Outer Sum Time,CSC Outer Sum Time,CSF2 Outer Sum Time,CSF3 Outer Sum Time,Armadillo Outer Sum Time,Eigen Sum Time,CSC Sum Time,CSF2 Sum Time,CSF3 Sum Time,Armadillo Sum Time,Eigen Norm Time,CSC Norm Time,CSF2 Norm Time,CSF3 Norm Time,Armadillo Norm Time,");
            fclose(file);
        }
    }

    // Destructor
    ~BenchAnalysis() {
        if (myMatrix != NULL)
            free(myMatrix);
    }

    void pushData(std::vector<uint64_t>& data) {

        // Constructor times
        EigenConstructorTimes.push_back(data.at(0));
        CSF1ConstructorTimes.push_back(data.at(1));
        CSF2ConstructorTimes.push_back(data.at(2));
        CSF3ConstructorTimes.push_back(data.at(3));
        ArmadilloConstructorTimes.push_back(data.at(4));

        // Inner iterator times
        EigenInnerIteratorTimes.push_back(data.at(5));
        CSF1InnerIteratorTimes.push_back(data.at(6));
        CSF2InnerIteratorTimes.push_back(data.at(7));
        CSF3InnerIteratorTimes.push_back(data.at(8));
        ArmadilloInnerIteratorTimes.push_back(data.at(9));

        // Scalar multiplication times
        EigenScalarMultiplicationTimes.push_back(data.at(10));
        CSF1ScalarMultiplicationTimes.push_back(data.at(11));
        CSF2ScalarMultiplicationTimes.push_back(data.at(12));
        CSF3ScalarMultiplicationTimes.push_back(data.at(13));
        ArmadilloScalarMultiplicationTimes.push_back(data.at(14));

        // Vector multiplication times
        EigenVectorMultiplicationTimes.push_back(data.at(15));
        CSF1VectorMultiplicationTimes.push_back(data.at(16));
        CSF2VectorMultiplicationTimes.push_back(data.at(17));
        CSF3VectorMultiplicationTimes.push_back(data.at(18));
        ArmadilloVectorMultiplicationTimes.push_back(data.at(19));

        // Memory usage
        EigenMemoryUsage.push_back(data.at(20));
        CSF1MemoryUsage.push_back(data.at(21));
        CSF2MemoryUsage.push_back(data.at(22));
        CSF3MemoryUsage.push_back(data.at(23));
        ArmadilloMemoryUsage.push_back(data.at(24));

        // Transpose times
        EigenTransposeTimes.push_back(data.at(25));
        CSF1TransposeTimes.push_back(data.at(26));
        CSF2TransposeTimes.push_back(data.at(27));
        CSF3TransposeTimes.push_back(data.at(28));
        ArmadilloTransposeTimes.push_back(data.at(29));

        // Matrix multiplication times
        EigenMatrixMultiplicationTimes.push_back(data.at(30));
        CSF1MatrixMultiplicationTimes.push_back(data.at(31));
        CSF2MatrixMultiplicationTimes.push_back(data.at(32));
        CSF3MatrixMultiplicationTimes.push_back(data.at(33));
        ArmadilloMatrixMultiplicationTimes.push_back(data.at(34));

        // Outer sum times
        EigenOuterSumTimes.push_back(data.at(35));
        CSF1OuterSumTimes.push_back(data.at(36));
        CSF2OuterSumTimes.push_back(data.at(37));
        CSF3OuterSumTimes.push_back(data.at(38));
        ArmadilloOuterSumTimes.push_back(data.at(39));

        // Sum times
        EigenSumTimes.push_back(data.at(40));
        CSF1SumTimes.push_back(data.at(41));
        CSF2SumTimes.push_back(data.at(42));
        CSF3SumTimes.push_back(data.at(43));
        ArmadilloSumTimes.push_back(data.at(44));

        // Norm times
        EigenNormTimes.push_back(data.at(45));
        CSF1NormTimes.push_back(data.at(46));
        CSF2NormTimes.push_back(data.at(47));
        CSF3NormTimes.push_back(data.at(48));
        ArmadilloNormTimes.push_back(data.at(49));

        FILE* file = fopen("rawTimeData.csv", "a");
        fprintf(file, "%f, %f, %f, %f, %f, %f,",
                myMatrix->matrixNum,
                myMatrix->matrixRows,
                myMatrix->matrixCols,
                myMatrix->matrixNonzeros,
                myMatrix->matrixRedundancy,
                myMatrix->matrixDensity);
        // Finally, print the data to the file
        for (uint32_t i = 0; i < data.size(); ++i) {
            fprintf(file, "%" PRIu64 ",", data.at(i));
        }
        fprintf(file, "\n");

        fclose(file);
    }

    /**
     * @brief Finds the average time in a vector
     *
     * @param times
     */

    uint64_t average(std::vector<uint64_t>& times) {
        uint64_t sum = 0;
        for (uint32_t i = 0; i < times.size(); i++) {
            sum += times.at(i);
        }
        return sum / times.size();
    }

    /**
     * @brief Finds the maximum value in a vector
     *
     * @tparam uint64_t
     * @param times
     */

    uint64_t max(std::vector<uint64_t> times) {
        uint64_t max = times.at(0);
        for (uint32_t i = 1; i < times.size(); i++) {
            if (times.at(i) > max) {
                max = times.at(i);
            }
        }
        return max;
    }

    uint64_t min(std::vector<uint64_t> times) {
        uint64_t min = times.at(0);
        for (uint32_t i = 1; i < times.size(); i++) {
            if (times.at(i) < min) {
                min = times.at(i);
            }
        }
        return min;
    }

    uint64_t median(std::vector<uint64_t> times) {
        std::sort(times.begin(), times.end());
        if (times.size() % 2 == 0) {
            return (times.at(times.size() / 2 - 1) + times.at(times.size() / 2)) / 2;
        }
        else {
            return times.at(times.size() / 2);
        }
    }

    uint64_t quarter1Percentile(std::vector<uint64_t> times) {
        std::sort(times.begin(), times.end());
        return times.at(times.size() / 4);
    }

    uint64_t quarter3Percentile(std::vector<uint64_t> times) {
        std::sort(times.begin(), times.end());
        return times.at(times.size() * 3 / 4);
    }

    void printTimesToCSV() {
        std::vector<uint64_t> data;

        // Average times
        // Constructors
        // print all elements of EigenConstructorTimes

        data.push_back(average(EigenConstructorTimes));
        data.push_back(average(CSF1ConstructorTimes));
        data.push_back(average(CSF2ConstructorTimes));
        data.push_back(average(CSF3ConstructorTimes));
        data.push_back(average(ArmadilloConstructorTimes));

        // Inner iterators
        data.push_back(average(EigenInnerIteratorTimes));
        data.push_back(average(CSF1InnerIteratorTimes));
        data.push_back(average(CSF2InnerIteratorTimes));
        data.push_back(average(CSF3InnerIteratorTimes));
        data.push_back(average(ArmadilloInnerIteratorTimes));

        // Scalar Multiplication
        data.push_back(average(EigenScalarMultiplicationTimes));
        data.push_back(average(CSF1ScalarMultiplicationTimes));
        data.push_back(average(CSF2ScalarMultiplicationTimes));
        data.push_back(average(CSF3ScalarMultiplicationTimes));
        data.push_back(average(ArmadilloScalarMultiplicationTimes));

        // Vector Multiplication
        data.push_back(average(EigenVectorMultiplicationTimes));
        data.push_back(average(CSF1VectorMultiplicationTimes));
        data.push_back(average(CSF2VectorMultiplicationTimes));
        data.push_back(average(CSF3VectorMultiplicationTimes));
        data.push_back(average(ArmadilloVectorMultiplicationTimes));

        // Memory usage
        data.push_back(average(EigenMemoryUsage));
        data.push_back(average(CSF1MemoryUsage));
        data.push_back(average(CSF2MemoryUsage));
        data.push_back(average(CSF3MemoryUsage));
        data.push_back(average(ArmadilloMemoryUsage));

        // Transpose times
        data.push_back(average(EigenTransposeTimes));
        data.push_back(average(CSF1TransposeTimes));
        data.push_back(average(CSF2TransposeTimes));
        data.push_back(average(CSF3TransposeTimes));
        data.push_back(average(ArmadilloTransposeTimes));

        // Matrix multiplication times
        data.push_back(average(EigenMatrixMultiplicationTimes));
        data.push_back(average(CSF1MatrixMultiplicationTimes));
        data.push_back(average(CSF2MatrixMultiplicationTimes));
        data.push_back(average(CSF3MatrixMultiplicationTimes));
        data.push_back(average(ArmadilloMatrixMultiplicationTimes));

        // Outer sum times
        data.push_back(average(EigenOuterSumTimes));
        data.push_back(average(CSF1OuterSumTimes));
        data.push_back(average(CSF2OuterSumTimes));
        data.push_back(average(CSF3OuterSumTimes));
        data.push_back(average(ArmadilloOuterSumTimes));

        // Sum times
        data.push_back(average(EigenSumTimes));
        data.push_back(average(CSF1SumTimes));
        data.push_back(average(CSF2SumTimes));
        data.push_back(average(CSF3SumTimes));
        data.push_back(average(ArmadilloSumTimes));

        // Norm times
        data.push_back(average(EigenNormTimes));
        data.push_back(average(CSF1NormTimes));
        data.push_back(average(CSF2NormTimes));
        data.push_back(average(CSF3NormTimes));
        data.push_back(average(ArmadilloNormTimes));

        // Mins
        // Constructors
        //  data.push_back(min(EigenConstructorTimes));
        //  data.push_back(min(CSF1ConstructorTimes));
        //  data.push_back(min(CSF2ConstructorTimes));
        //  data.push_back(min(CSF3ConstructorTimes));
        //  data.push_back(min(ArmadilloConstructorTimes));

        // //Inner iterators
        // data.push_back(min(EigenInnerIteratorTimes));
        // data.push_back(min(CSF1InnerIteratorTimes)
        // data.push_back(min(CSF2InnerIteratorTimes));
        // data.push_back(min(CSF3InnerIteratorTimes));
        // data.push_back(min(ArmadilloInnerIteratorTimes));

        // //Scalar Multiplication
        // data.push_back(min(EigenScalarMultiplicationTimes));
        // data.push_back(min(CSF1ScalarMultiplicationTimes)
        // data.push_back(min(CSF2ScalarMultiplicationTimes));
        // data.push_back(min(CSF3ScalarMultiplicationTimes));
        // data.push_back(min(ArmadilloScalarMultiplicationTimes));

        // //Vector Multiplication
        // data.push_back(min(EigenVectorMultiplicationTimes));
        // data.push_back(min(CSF2VectorMultiplicationTimes));
        // data.push_back(min(CSF3VectorMultiplicationTimes));
        // data.push_back(min(ArmadilloVectorMultiplicationTimes));

        // // Memory usage
        // data.push_back(min(EigenMemoryUsage));
        // data.push_back(min(CSF2MemoryUsage));
        // data.push_back(min(CSF3MemoryUsage));
        // data.push_back(min(ArmadilloMemoryUsage));

        // // Transpose times
        // data.push_back(min(EigenTransposeTimes));
        // data.push_back(min(CSF2TransposeTimes));
        // data.push_back(min(CSF3TransposeTimes));
        // data.push_back(min(ArmadilloTransposeTimes));

        // // Matrix multiplication times
        // data.push_back(min(EigenMatrixMultiplicationTimes));
        // data.push_back(min(CSF2MatrixMultiplicationTimes));
        // data.push_back(min(CSF3MatrixMultiplicationTimes));
        // data.push_back(min(ArmadilloMatrixMultiplicationTimes));

        // //Outer sum times
        // data.push_back(min(EigenOuterSumTimes));
        // data.push_back(min(CSF2OuterSumTimes));
        // data.push_back(min(CSF3OuterSumTimes));
        // data.push_back(min(ArmadilloOuterSumTimes));

        // //Sum times
        // data.push_back(min(EigenSumTimes));
        // data.push_back(min(CSF2SumTimes));
        // data.push_back(min(CSF3SumTimes));
        // data.push_back(min(ArmadilloSumTimes));

        // //Norm times
        // data.push_back(min(EigenNormTimes));
        // data.push_back(min(CSF2NormTimes));
        // data.push_back(min(CSF3NormTimes));
        // data.push_back(min(ArmadilloNormTimes));

        // //Quartile 1

        // //Constructors
        // data.push_back(quarter1Percentile(EigenConstructorTimes));
        // data.push_back(quarter1Percentile(CSF2ConstructorTimes));
        // data.push_back(quarter1Percentile(CSF3ConstructorTimes));
        // data.push_back(quarter1Percentile(ArmadilloConstructorTimes));

        // //Inner iterators
        // data.push_back(quarter1Percentile(EigenInnerIteratorTimes));
        // data.push_back(quarter1Percentile(CSF2InnerIteratorTimes));
        // data.push_back(quarter1Percentile(CSF3InnerIteratorTimes));
        // data.push_back(quarter1Percentile(ArmadilloInnerIteratorTimes));

        // //Scalar Multiplication
        // data.push_back(quarter1Percentile(EigenScalarMultiplicationTimes));
        // data.push_back(quarter1Percentile(CSF2ScalarMultiplicationTimes));
        // data.push_back(quarter1Percentile(CSF3ScalarMultiplicationTimes));
        // data.push_back(quarter1Percentile(ArmadilloScalarMultiplicationTimes));

        // //Vector Multiplication
        // data.push_back(quarter1Percentile(EigenVectorMultiplicationTimes));
        // data.push_back(quarter1Percentile(CSF2VectorMultiplicationTimes));
        // data.push_back(quarter1Percentile(CSF3VectorMultiplicationTimes));
        // data.push_back(quarter1Percentile(ArmadilloVectorMultiplicationTimes));

        // // Memory usage
        // data.push_back(quarter1Percentile(EigenMemoryUsage));
        // data.push_back(quarter1Percentile(CSF2MemoryUsage));
        // data.push_back(quarter1Percentile(CSF3MemoryUsage));
        // data.push_back(quarter1Percentile(ArmadilloMemoryUsage));

        // // Transpose times
        // data.push_back(quarter1Percentile(EigenTransposeTimes));
        // data.push_back(quarter1Percentile(CSF2TransposeTimes));
        // data.push_back(quarter1Percentile(CSF3TransposeTimes));
        // data.push_back(quarter1Percentile(ArmadilloTransposeTimes));

        // // Matrix multiplication times
        // data.push_back(quarter1Percentile(EigenMatrixMultiplicationTimes));
        // data.push_back(quarter1Percentile(CSF2MatrixMultiplicationTimes));
        // data.push_back(quarter1Percentile(CSF3MatrixMultiplicationTimes));
        // data.push_back(quarter1Percentile(ArmadilloMatrixMultiplicationTimes));

        // //Outer sum times
        // data.push_back(quarter1Percentile(EigenOuterSumTimes));
        // data.push_back(quarter1Percentile(CSF2OuterSumTimes));
        // data.push_back(quarter1Percentile(CSF3OuterSumTimes));
        // data.push_back(quarter1Percentile(ArmadilloOuterSumTimes));

        // //Sum times
        // data.push_back(quarter1Percentile(EigenSumTimes));
        // data.push_back(quarter1Percentile(CSF2SumTimes));
        // data.push_back(quarter1Percentile(CSF3SumTimes));
        // data.push_back(quarter1Percentile(ArmadilloSumTimes));

        // //Norm times
        // data.push_back(quarter1Percentile(EigenNormTimes));
        // data.push_back(quarter1Percentile(CSF2NormTimes));
        // data.push_back(quarter1Percentile(CSF3NormTimes));
        // data.push_back(quarter1Percentile(ArmadilloNormTimes));

        // //Medians
        // //Constructors
        // data.push_back(median(EigenConstructorTimes));
        // data.push_back(median(CSF2ConstructorTimes));
        // data.push_back(median(CSF3ConstructorTimes));
        // data.push_back(median(ArmadilloConstructorTimes));

        // //Inner iterators
        // data.push_back(median(EigenInnerIteratorTimes));
        // data.push_back(median(CSF2InnerIteratorTimes));
        // data.push_back(median(CSF3InnerIteratorTimes));
        // data.push_back(median(ArmadilloInnerIteratorTimes));

        // //Scalar Multiplication
        // data.push_back(median(EigenScalarMultiplicationTimes));
        // data.push_back(median(CSF2ScalarMultiplicationTimes));
        // data.push_back(median(CSF3ScalarMultiplicationTimes));
        // data.push_back(median(ArmadilloScalarMultiplicationTimes));

        // //Vector Multiplication
        // data.push_back(median(EigenVectorMultiplicationTimes));
        // data.push_back(median(CSF2VectorMultiplicationTimes));
        // data.push_back(median(CSF3VectorMultiplicationTimes));
        // data.push_back(median(ArmadilloVectorMultiplicationTimes));

        // // Memory usage
        // data.push_back(median(EigenMemoryUsage));
        // data.push_back(median(CSF2MemoryUsage));
        // data.push_back(median(CSF3MemoryUsage));
        // data.push_back(median(ArmadilloMemoryUsage));

        // // Transpose times
        // data.push_back(median(EigenTransposeTimes));
        // data.push_back(median(CSF2TransposeTimes));
        // data.push_back(median(CSF3TransposeTimes));
        // data.push_back(median(ArmadilloTransposeTimes));

        // // Matrix multiplication times
        // data.push_back(median(EigenMatrixMultiplicationTimes));
        // data.push_back(median(CSF2MatrixMultiplicationTimes));
        // data.push_back(median(CSF3MatrixMultiplicationTimes));
        // data.push_back(median(ArmadilloMatrixMultiplicationTimes));

        // //Outer sum times
        // data.push_back(median(EigenOuterSumTimes));
        // data.push_back(median(CSF2OuterSumTimes));
        // data.push_back(median(CSF3OuterSumTimes));
        // data.push_back(median(ArmadilloOuterSumTimes));

        // //Sum times
        // data.push_back(median(EigenSumTimes));
        // data.push_back(median(CSF2SumTimes));
        // data.push_back(median(CSF3SumTimes));
        // data.push_back(median(ArmadilloSumTimes));

        // //Norm times
        // data.push_back(median(EigenNormTimes));
        // data.push_back(median(CSF2NormTimes));
        // data.push_back(median(CSF3NormTimes));
        // data.push_back(median(ArmadilloNormTimes));

        // //Quartile 3
        // //Constructors
        // data.push_back(quarter3Percentile(EigenConstructorTimes));
        // data.push_back(quarter3Percentile(CSF2ConstructorTimes));
        // data.push_back(quarter3Percentile(CSF3ConstructorTimes));
        // data.push_back(quarter3Percentile(ArmadilloConstructorTimes));

        // //Inner iterators
        // data.push_back(quarter3Percentile(EigenInnerIteratorTimes));
        // data.push_back(quarter3Percentile(CSF2InnerIteratorTimes));
        // data.push_back(quarter3Percentile(CSF3InnerIteratorTimes));
        // data.push_back(quarter3Percentile(ArmadilloInnerIteratorTimes));

        // //Scalar Multiplication
        // data.push_back(quarter3Percentile(EigenScalarMultiplicationTimes));
        // data.push_back(quarter3Percentile(CSF2ScalarMultiplicationTimes));
        // data.push_back(quarter3Percentile(CSF3ScalarMultiplicationTimes));
        // data.push_back(quarter3Percentile(ArmadilloScalarMultiplicationTimes));

        // //Vector Multiplication
        // data.push_back(quarter3Percentile(EigenVectorMultiplicationTimes));
        // data.push_back(quarter3Percentile(CSF2VectorMultiplicationTimes));
        // data.push_back(quarter3Percentile(CSF3VectorMultiplicationTimes));
        // data.push_back(quarter3Percentile(ArmadilloVectorMultiplicationTimes));

        // // Memory usage
        // data.push_back(quarter3Percentile(EigenMemoryUsage));
        // data.push_back(quarter3Percentile(CSF2MemoryUsage));
        // data.push_back(quarter3Percentile(CSF3MemoryUsage));
        // data.push_back(quarter3Percentile(ArmadilloMemoryUsage));

        // // Transpose times
        // data.push_back(quarter3Percentile(EigenTransposeTimes));
        // data.push_back(quarter3Percentile(CSF2TransposeTimes));
        // data.push_back(quarter3Percentile(CSF3TransposeTimes));
        // data.push_back(quarter3Percentile(ArmadilloTransposeTimes));

        // // Matrix multiplication times
        // data.push_back(quarter3Percentile(EigenMatrixMultiplicationTimes));
        // data.push_back(quarter3Percentile(CSF2MatrixMultiplicationTimes));
        // data.push_back(quarter3Percentile(CSF3MatrixMultiplicationTimes));
        // data.push_back(quarter3Percentile(ArmadilloMatrixMultiplicationTimes));

        // //Outer sum times
        // data.push_back(quarter3Percentile(EigenOuterSumTimes));
        // data.push_back(quarter3Percentile(CSF2OuterSumTimes));
        // data.push_back(quarter3Percentile(CSF3OuterSumTimes));
        // data.push_back(quarter3Percentile(ArmadilloOuterSumTimes));

        // //Sum times
        // data.push_back(quarter3Percentile(EigenSumTimes));
        // data.push_back(quarter3Percentile(CSF2SumTimes));
        // data.push_back(quarter3Percentile(CSF3SumTimes));
        // data.push_back(quarter3Percentile(ArmadilloSumTimes));

        // //Norm times
        // data.push_back(quarter3Percentile(EigenNormTimes));
        // data.push_back(quarter3Percentile(CSF2NormTimes));
        // data.push_back(quarter3Percentile(CSF3NormTimes));
        // data.push_back(quarter3Percentile(ArmadilloNormTimes));

        // // Max times
        // //Constructors
        // data.push_back(max(EigenConstructorTimes));
        // data.push_back(max(CSF2ConstructorTimes));
        // data.push_back(max(CSF3ConstructorTimes));
        // data.push_back(max(ArmadilloConstructorTimes));

        // //Inner iterators
        // data.push_back(max(EigenInnerIteratorTimes));
        // data.push_back(max(CSF2InnerIteratorTimes));
        // data.push_back(max(CSF3InnerIteratorTimes));
        // data.push_back(max(ArmadilloInnerIteratorTimes));

        // //Scalar Multiplication
        // data.push_back(max(EigenScalarMultiplicationTimes));
        // data.push_back(max(CSF2ScalarMultiplicationTimes));
        // data.push_back(max(CSF3ScalarMultiplicationTimes));
        // data.push_back(max(ArmadilloScalarMultiplicationTimes));

        // //Vector Multiplication
        // data.push_back(max(EigenVectorMultiplicationTimes));
        // data.push_back(max(CSF2VectorMultiplicationTimes));
        // data.push_back(max(CSF3VectorMultiplicationTimes));
        // data.push_back(max(ArmadilloVectorMultiplicationTimes));

        // // Memory usage
        // data.push_back(max(EigenMemoryUsage));
        // data.push_back(max(CSF2MemoryUsage));
        // data.push_back(max(CSF3MemoryUsage));
        // data.push_back(max(ArmadilloMemoryUsage));

        // // Transpose times
        // data.push_back(max(EigenTransposeTimes));
        // data.push_back(max(CSF2TransposeTimes));
        // data.push_back(max(CSF3TransposeTimes));
        // data.push_back(max(ArmadilloTransposeTimes));

        // // Matrix multiplication times
        // data.push_back(max(EigenMatrixMultiplicationTimes));
        // data.push_back(max(CSF2MatrixMultiplicationTimes));
        // data.push_back(max(CSF3MatrixMultiplicationTimes));
        // data.push_back(max(ArmadilloMatrixMultiplicationTimes));

        // //Outer sum times
        // data.push_back(max(EigenOuterSumTimes));
        // data.push_back(max(CSF2OuterSumTimes));
        // data.push_back(max(CSF3OuterSumTimes));
        // data.push_back(max(ArmadilloOuterSumTimes));

        // //Sum times
        // data.push_back(max(EigenSumTimes));
        // data.push_back(max(CSF2SumTimes));
        // data.push_back(max(CSF3SumTimes));
        // data.push_back(max(ArmadilloSumTimes));

        // //Norm times
        // data.push_back(max(EigenNormTimes));
        // data.push_back(max(CSF2NormTimes));
        // data.push_back(max(CSF3NormTimes));
        // data.push_back(max(ArmadilloNormTimes));

        /*
                                                                 ORDER: Eigen, CSF2, CSF3, Armadillo
            ||
                              ATTRIBUTES                      ||                                          AVERAGE                                              ||                                             MAX                                                |
            ID | Rows | Cols | Nonzeros | Redundancy, Density || Constructor | Iterator | scalar multiplcation | vector multiplication | matrix multiplication ||  Constructor | Iterator | scalar multiplcation | vector multiplication | matrix multiplication |
        */

        FILE* file = fopen("timeData.csv", "a");
        fprintf(file, "%f, %f, %f, %f, %f, %f,",
                myMatrix->matrixNum,
                myMatrix->matrixRows,
                myMatrix->matrixCols,
                myMatrix->matrixNonzeros,
                myMatrix->matrixRedundancy,
                myMatrix->matrixDensity);
        // Finally, print the data to the file
        for (uint32_t i = 0; i < data.size(); ++i) {
            fprintf(file, "%" PRIu64 ",", data.at(i));
        }
        fprintf(file, "\n");
        fclose(file);
    }
};

/**
 * This is for future referance if we need to edit headers
 */

 /**
  *        "Matrix Number,
             Matrix Rows,
             Matrix Cols,
             Matrix Nonzeros,
             Matrix Redundancy,
             Matrix Density,
             Avg Eigen Constructor Time,
             Avg CSC Constructor Time,
             Avg CSF2 Constructor Time,
             Avg CSF3 Constructor Time,
             Avg Armadillo Constructor Time,
             Avg Eigen InnerIterator Time,
             Avg CSC InnerIterator Time,
             Avg CSF2 InnerIterator Time,
             Avg CSF3 InnerIterator Time,
             Avg Armadillo InnerIterator Time,
             Avg Eigen Scalar Multiplication Time,
             Avg CSC Scalar Multiplication Time,
             Avg CSF2 Scalar Multiplication Time,
             Avg CSF3 Scalar Multiplication Time,
             Avg Armadillo Scalar Multiplication Time,
             Avg Eigen Vector Multiplication Time,
             Avg CSC Vector Multiplication Time,
             Avg CSF2 Vector Multiplication Time,
             Avg CSF3 Vector Multiplication Time,
             Avg Armadillo Vector Multiplication Time,
             Avg Eigen Memory Usage,
             Avg CSC Memory Usage,
             Avg CSF2 Memory Usage,
             Avg CSF3 Memory Usage,
             Avg Armadillo Memory Usage,
             Avg Eigen Transpose Time,
             Avg CSC Transpose Time,
             Avg CSF2 Transpose Time,
             Avg CSF3 Transpose Time,
             Avg Armadillo Transpose Time,
             Avg Eigen Matrix Multiplication Time,
             Avg CSC Matrix Multiplication Time,
             Avg CSF2 Matrix Multiplication Time,
             Avg CSF3 Matrix Multiplication Time,
             Avg Armadillo Matrix Multiplication Time,
             Avg Eigen Outer Sum Time,
             Avg CSC Outer Sum Time,
             Avg CSF2 Outer Sum Time,
             Avg CSF3 Outer Sum Time,
             Avg Armadillo Outer Sum Time,
             Avg Eigen Sum Time,
             Avg CSC Sum Time,
             Avg CSF2 Sum Time,
             Avg CSF3 Sum Time,
             Avg Armadillo Sum Time,
             Avg Eigen Norm Time,
             Avg CSC Norm Time,
             Avg CSF2 Norm Time,
             Avg CSF3 Norm Time,
             Avg Armadillo Norm Time"



 "Matrix Number,
     Matrix Rows,
     Matrix Cols,
     Matrix Nonzeros,
     Matrix Redundancy,
     Matrix Density,
     Eigen Constructor Time,
     CSC Constructor Time,
     CSF2 Constructor Time,
     CSF3 Constructor Time,
     Armadillo Constructor Time,
     Eigen InnerIterator Time,
     CSC InnerIterator Time,
     CSF2 InnerIterator Time,
     CSF3 InnerIterator Time,
     Armadillo InnerIterator Time,
     Eigen Scalar Multiplication Time,
     CSC Scalar Multiplication Time,
     CSF2 Scalar Multiplication Time,
     CSF3 Scalar Multiplication Time,
     Armadillo Scalar Multiplication Time,
     Eigen Vector Multiplication Time,
     CSC Vector Multiplication Time,
     CSF2 Vector Multiplication Time,
     CSF3 Vector Multiplication Time,
     Armadillo Vector Multiplication Time,
     Eigen Memory Usage,
     CSC Memory Usage,
     CSF2 Memory Usage,
     CSF3 Memory Usage,
     Armadillo Memory Usage,
     Eigen Transpose Time,
     CSC Transpose Time,
     CSF2 Transpose Time,
     CSF3 Transpose Time,
     Armadillo Transpose Time,
     Eigen Matrix Multiplication Time,
     CSC Matrix Multiplication Time,
     CSF2 Matrix Multiplication Time,
     CSF3 Matrix Multiplication Time,
     Armadillo Matrix Multiplication Time,
     Eigen Outer Sum Time,
     CSC Outer Sum Time,
     CSF2 Outer Sum Time,
     CSF3 Outer Sum Time,
     Armadillo Outer Sum Time,
     Eigen Sum Time,
     CSC Sum Time,
     CSF2 Sum Time,
     CSF3 Sum Time,
     Armadillo Sum Time,
     Eigen Norm Time,
     CSC Norm Time,
     CSF2 Norm Time,
     CSF3 Norm Time,
     Armadillo Norm Time,"
 */

 /*
            "Matrix Number,
             Matrix Rows,
             Matrix Cols,
             Matrix Nonzeros,
             Matrix Redundancy,
             Matrix Density,
             Avg Eigen Constructor Time,
             Avg CSF2 Constructor Time,
             Avg CSF3 Constructor Time,
             Avg Armadillo Constructor Time,
             Avg Eigen InnerIterator Time,
             Avg CSF2 InnerIterator Time,
             Avg CSF3 InnerIterator Time,
             Avg Armadillo InnerIterator Time,
             Avg Eigen Scalar Multiplication Time,
             Avg CSF2 Scalar Multiplication Time,
             Avg CSF3 Scalar Multiplication Time,
             Avg Armadillo Scalar Multiplication Time,
             Avg Eigen Vector Multiplication Time,
             Avg CSF2 Vector Multiplication Time,
             Avg CSF3 Vector Multiplication Time,
             Avg Armadillo Vector Multiplication Time,
             Avg Eigen Memory Usage,
             Avg CSF2 Memory Usage,
             Avg CSF3 Memory Usage,
             Avg Armadillo Memory Usage,
             Avg Eigen Transpose Time,
             Avg CSF2 Transpose Time,
             Avg CSF3 Transpose Time,
             Avg Armadillo Transpose Time,
             Avg Eigen Matrix Multiplication Time,
             Avg CSF2 Matrix Multiplication Time,
             Avg CSF3 Matrix Multiplication Time,
             Avg Armadillo Matrix Multiplication Time,
             Avg Eigen Outer Sum Time,
             Avg CSF2 Outer Sum Time,
             Avg CSF3 Outer Sum Time,
             Avg Armadillo Outer Sum Time,
             Avg Eigen Sum Time,
             Avg CSF2 Sum Time,
             Avg CSF3 Sum Time,
             Avg Armadillo Sum Time,
             Avg Eigen Norm Time,
             Avg CSF2 Norm Time,
             Avg CSF3 Norm Time,
             Avg Armadillo Norm Time,
             Min Eigen Constructor Time,
             Min CSF2 Constructor Time,
             Min CSF3 Constructor Time,
             Min Armadillo Constructor Time,
             Min Eigen InnerIterator Time,
             Min CSF2 InnerIterator Time,
             Min CSF3 InnerIterator Time,
             Min Armadillo InnerIterator Time,
             Min Eigen Scalar Multiplication Time,
             Min CSF2 Scalar Multiplication Time,
             Min CSF3 Scalar Multiplication Time,
             Min Armadillo Scalar Multiplication Time,
             Min Eigen Vector Multiplication Time,
             Min CSF2 Vector Multiplication Time,
             Min CSF3 Vector Multiplication Time,
             Min Armadillo Vector Multiplication Time,
             Min Eigen Memory Usage,
             Min CSF2 Memory Usage,
             Min CSF3 Memory Usage,
             Min Armadillo Memory Usage,
             Min Eigen Transpose Time,
             Min CSF2 Transpose Time,
             Min CSF3 Transpose Time,
             Min Armadillo Transpose Time,
             Min Eigen Matrix Multiplication Time,
             Min CSF2 Matrix Multiplication Time,
             Min CSF3 Matrix Multiplication Time,
             Min Armadillo Matrix Multiplication Time,
             Min Eigen Outer Sum Time,
             Min CSF2 Outer Sum Time,
             Min CSF3 Outer Sum Time,
             Min Armadillo Outer Sum Time,
             Min Eigen Sum Time,
             Min CSF2 Sum Time,
             Min CSF3 Sum Time,
             Min Armadillo Sum Time,
             Min Eigen Norm Time,
             Min CSF2 Norm Time,
             Min CSF3 Norm Time,
             Min Armadillo Norm Time,
             Q1 Eigen Constructor Time,
             Q1 CSF2 Constructor Time,
             Q1 CSF3 Constructor Time,
             Q1 Armadillo Constructor Time,
             Q1 Eigen InnerIterator Time,
             Q1 CSF2 InnerIterator Time,
             Q1 CSF3 InnerIterator Time,
             Q1 Armadillo InnerIterator Time,
             Q1 Eigen Scalar Multiplication Time,
             Q1 CSF2 Scalar Multiplication Time,
             Q1 CSF3 Scalar Multiplication Time,
             Q1 Armadillo Scalar Multiplication Time,
             Q1 Eigen Vector Multiplication Time,
             Q1 CSF2 Vector Multiplication Time,
             Q1 CSF3 Vector Multiplication Time,
             Q1 Armadillo Vector Multiplication Time,
             Q1 Eigen Memory Usage,
             Q1 CSF2 Memory Usage,
             Q1 CSF3 Memory Usage,
             Q1 Armadillo Memory Usage,
             Q1 Eigen Transpose Time,
             Q1 CSF2 Transpose Time,
             Q1 CSF3 Transpose Time,
             Q1 Armadillo Transpose Time,
             Q1 Eigen Matrix Multiplication Time,
             Q1 CSF2 Matrix Multiplication Time,
             Q1 CSF3 Matrix Multiplication Time,
             Q1 Armadillo Matrix Multiplication Time,
             Q1 Eigen Outer Sum Time,
             Q1 CSF2 Outer Sum Time,
             Q1 CSF3 Outer Sum Time,
             Q1 Armadillo Outer Sum Time,
             Q1 Eigen Sum Time,
             Q1 CSF2 Sum Time,
             Q1 CSF3 Sum Time,
             Q1 Armadillo Sum Time,
             Q1 Eigen Norm Time,
             Q1 CSF2 Norm Time,
             Q1 CSF3 Norm Time,
             Q1 Armadillo Norm Time,
             Median Eigen Constructor Time,
             Median CSF2 Constructor Time,
             Median CSF3 Constructor Time,
             Median Armadillo Constructor Time,
             Median Eigen InnerIterator Time,
             Median CSF2 InnerIterator Time,
             Median CSF3 InnerIterator Time,
             Median Armadillo InnerIterator Time,
             Median Eigen Scalar Multiplication Time,
             Median CSF2 Scalar Multiplication Time,
             Median CSF3 Scalar Multiplication Time,
             Median Armadillo Scalar Multiplication Time,
             Median Eigen Vector Multiplication Time,
             Median CSF2 Vector Multiplication Time,
             Median CSF3 Vector Multiplication Time,
             Median Armadillo Vector Multiplication Time,
             Median Eigen Memory Usage,
             Median CSF2 Memory Usage,
             Median CSF3 Memory Usage,
             Median Armadillo Memory Usage,
             Median Eigen Transpose Time,
             Median CSF2 Transpose Time,
             Median CSF3 Transpose Time,
             Median Armadillo Transpose Time,
             Median Eigen Matrix Multiplication Time,
             Median CSF2 Matrix Multiplication Time,
             Median CSF3 Matrix Multiplication Time,
             Median Armadillo Matrix Multiplication Time,
             Median Eigen Outer Sum Time,
             Median CSF2 Outer Sum Time,
             Median CSF3 Outer Sum Time,
             Median Armadillo Outer Sum Time,
             Median Eigen Sum Time,
             Median CSF2 Sum Time,
             Median CSF3 Sum Time,
             Median Armadillo Sum Time,
             Median Eigen Norm Time,
             Median CSF2 Norm Time,
             Median CSF3 Norm Time,
             Median Armadillo Norm Time,
             Q3 Eigen Constructor Time,
             Q3 CSF2 Constructor Time,
             Q3 CSF3 Constructor Time,
             Q3 Armadillo Constructor Time,
             Q3 Eigen InnerIterator Time,
             Q3 CSF2 InnerIterator Time,
             Q3 CSF3 InnerIterator Time,
             Q3 Armadillo InnerIterator Time,
             Q3 Eigen Scalar Multiplication,
             Q3 CSF2 Scalar Multiplication Time,
             Q3 CSF3 Scalar Multiplication Time,
             Q3 Armadillo Scalar Multiplication Time,
             Q3 Eigen Vector Multiplication Time,
             Q3 CSF2 Vector Multiplication Time,
             Q3 CSF3 Vector Multiplication Time,
             Q3 Armadillo Vector Multiplication Time,
             Q3 Eigen Memory Usage,
             Q3 CSF2 Memory Usage,
             Q3 CSF3 Memory Usage,
             Q3 Armadillo Memory Usage,
             Q3 Eigen Transpose Time,
             Q3 CSF2 Transpose Time,
             Q3 CSF3 Transpose Time,
             Q3 Armadillo Transpose Time,
             Q3 Eigen Matrix Multiplication Time,
             Q3 CSF2 Matrix Multiplication Time,
             Q3 CSF3 Matrix Multiplication Time,
             Q3 Armadillo Matrix Multiplication Time,
             Q3 Eigen Outer Sum Time,
             Q3 CSF2 Outer Sum Time,
             Q3 CSF3 Outer Sum Time,
             Q3 Armadillo Outer Sum Time,
             Q3 Eigen Sum Time,
             Q3 CSF2 Sum Time,
             Q3 CSF3 Sum Time,
             Q3 Armadillo Sum Time,
             Q3 Eigen Norm Time,
             Q3 CSF2 Norm Time,
             Q3 CSF3 Norm Time,
             Q3 Armadillo Norm Time,
             Max Eigen Constructor Time,
             Max CSF2 Constructor Time,
             Max CSF3 Constructor Time,
             Max Armadillo Constructor Time,
             Max Eigen InnerIterator Time,
             Max CSF2 InnerIterator Time,
             Max CSF3 InnerIterator Time,
             Max Armadillo InnerIterator Time,
             Max Eigen Scalar Multiplication Time,
             Max CSF2 Scalar Multiplication Time,
             Max CSF3 Scalar Multiplication Time,
             Max Armadillo Scalar Multiplication Time,
             Max Eigen Vector Multiplication Time,
             Max CSF2 Vector Multiplication Time,
             Max CSF3 Vector Multiplication Time,
             Max Armadillo Vector Multiplication Time,
             Max Eigen Memory Usage,
             Max CSF2 Memory Usage,
             Max CSF3 Memory Usage,
             Max Armadillo Memory Usage,
             Max Eigen Transpose Time,
             Max CSF2 Transpose Time,
             Max CSF3 Transpose Time,
             Max Armadillo Transpose Time,
             Max Eigen Matrix Multiplication Time,
             Max CSF2 Matrix Multiplication Time,
             Max CSF3 Matrix Multiplication Time
             Max Armadillo Matrix Multiplication Time.
             Max Eigen Outer Sum Time,
             Max CSF2 Outer Sum Time,
             Max CSF3 Outer Sum Time,
             Max Armadillo Outer Sum Time,
             Max Eigen Sum Time,
             Max CSF2 Sum Time,
             Max CSF3 Sum Time,
             Max Armadillo Sum Time,
             Max Eigen Norm Time,
             Max CSF2 Norm Time,
             Max CSF3 Norm Time,
             Max Armadillo Norm Time"
 */
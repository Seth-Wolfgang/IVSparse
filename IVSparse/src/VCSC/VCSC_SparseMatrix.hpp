/**
 * @file VCSC_SparseMatrix.hpp
 * @author Skyler Ruiter and Seth Wolfgang
 * @brief Header File for VCSC Sparse Matrix Declarations
 * @version 0.1
 * @date 2023-07-03
 */

#pragma once

namespace IVSparse {

    /**
     * The VCSC Sparse Matrix Class is a version of the CSC format. The difference
     * is that CSC is base CSC but VCSC is CSC with run length encoding and a vector
     * that keeps track of the number of occurances of a value in a column. This
     * allows for fast level 1 BLAS operations and still compresses more with
     * redundant data than CSC.
     */
    template <typename T, typename indexT, bool columnMajor>
    class SparseMatrix<T, indexT, 2, columnMajor> : public SparseMatrixBase {
        private:
        //* The Matrix Data *//

        T** values = nullptr;        // The values of the matrix
        indexT** counts = nullptr;   // The counts of the matrix
        indexT** indices = nullptr;  // The indices of the matrix

        indexT* valueSizes = nullptr;  // The sizes of the value arrays
        indexT* indexSizes = nullptr;  // The sizes of the index arrays

        //* Private Methods *//

        // Compression Algorithm for going from CSC to VCSC or IVCSCC
        template <typename T2, typename indexT2>
        void compressCSC(T2* vals, indexT2* innerIndices, indexT2* outerPointers);

        // Encodes the value type of the matrix
        void encodeValueType();

        // Checks if the value type is correct for the matrix
        void checkValueType();

        // performs some simple user checks on the matrices metadata
        void userChecks();

        // Calculates the current byte size of the matrix in memory
        void calculateCompSize();

        // Private Helper Constructor for tranposing a IVSparse matrix
        SparseMatrix(std::unordered_map<T, std::vector<indexT>> maps[],
                     uint32_t num_rows, uint32_t num_cols);

        // Scalar Multiplication
        inline IVSparse::SparseMatrix<T, indexT, 2, columnMajor> scalarMultiply(
            T scalar);

        // In Place Scalar Multiplication
        inline void inPlaceScalarMultiply(T scalar);

        // Matrix Vector Multiplication
        inline Eigen::Matrix<T, -1, 1> vectorMultiply(Eigen::Matrix<T, -1, 1>& vec);

        // Matrix Vector Multiplication 2 (with IVSparse Vector)
        inline Eigen::Matrix<T, -1, 1> vectorMultiply(
            typename SparseMatrix<T, indexT, 2, columnMajor>::Vector& vec);

        public:
        //* Nested Subclasses *//

        // The Vector Class for VCSC Matrices
        class Vector;

        // The Iterator Class for VCSC Matrices
        class InnerIterator;

        //* Constructors and Destructor *//
        /** @name Constructors
         */
         ///@{

         /**
          * Construct an empty IVSparse matrix \n \n
          * The matrix will have 0 rows and 0 columns and
          * will not be initialized with any values. All data
          * will be set to nullptr.
          *
          * @attention This constructor is not recommended for use as updating a
          * IVSparse matrix is not well supported.
          */
        SparseMatrix() {};

        /**
         * @param mat The Eigen Sparse Matrix to be compressed
         *
         * Eigen Sparse Matrix Constructor \n \n
         * This constructor takes an Eigen Sparse Matrix and compresses it into a
         * IVSparse matrix.
         */
        SparseMatrix(Eigen::SparseMatrix<T>& mat);

        /**
         * @param mat The Eigen Sparse Matrix to be compressed
         *
         * Eigen Sparse Matrix Constructor (Row Major) \n \n
         * Same as previous constructor but for Row Major Eigen Sparse Matrices.
         */
        SparseMatrix(Eigen::SparseMatrix<T, Eigen::RowMajor>& mat);

        /**
         * @tparam compressionLevel2 The compression level of the IVSparse matrix to
         * convert
         * @param mat The IVSparse matrix to convert
         *
         * Convert a IVSparse matrix of a different compression level to this
         * compression level. \n \n This constructor takes in a IVSparse matrix of the
         * same storage order, value, and index type and converts it to a different
         * compresion level. This is useful for converting between compression levels
         * without having to go through the CSC format.
         */
        template <uint8_t compressionLevel2>
        SparseMatrix(
            IVSparse::SparseMatrix<T, indexT, compressionLevel2, columnMajor>& other);

        /**
         * @param other The IVSparse matrix to be copied
         *
         * Deep Copy Constructor \n \n
         * This constructor takes in a VCSC matrix and creates a deep copy of it.
         */
        SparseMatrix(const IVSparse::SparseMatrix<T, indexT, 2, columnMajor>& other);

        /**
         * Raw CSC Constructor \n \n
         * This constructor takes in raw CSC storage format pointers and converts it
         * to a VCSC matrix. One could also take this information and convert to an
         * Eigen Sparse Matrix and then to a VCSC matrix.
         */
        template <typename T2, typename indexT2>
        SparseMatrix(T2* vals, indexT2* innerIndices, indexT2* outerPtr,
                     uint32_t num_rows, uint32_t num_cols, uint32_t nnz);

        /**
         * COO Tuples Constructor \n \n
         * This constructor takes in a list of tuples in COO format which can be
         * unsorted but without duplicates. The tuples are sorted and then converted
         * to a IVSparse matrix.
         *
         * @note COO is (row, col, value) format.
         */
        template <typename T2, typename indexT2>
        SparseMatrix(std::vector<std::tuple<indexT2, indexT2, T2>>& entries,
                     uint64_t num_rows, uint32_t num_cols, uint32_t nnz);

        /**
         * @param vec The vector to construct the matrix from
         *
         * IVSparse Vector Constructor \n \n
         * This constructor takes in a single VCSC vector and creates a one column/row
         * VCSC matrix.
         */
        SparseMatrix(
            typename IVSparse::SparseMatrix<T, indexT, 2, columnMajor>::Vector& vec);

        /**
         * @param vecs The vector of VCSC vectors to construct from.
         *
         * Vector of IVSparse Vectors Constructor \n \n
         * This constructor takes in an vector of VCSC vectors and creates a VCSC
         * matrix from them.
         */
        SparseMatrix(std::vector<typename IVSparse::SparseMatrix<
                     T, indexT, 2, columnMajor>::Vector>& vecs);

        /**
         * @param filename The filepath of the matrix to be read in
         *
         * File Constructor \n \n
         * Given a filepath to a VCSC matrix written to file this constructor will
         * read in the matrix and construct it.
         */
        SparseMatrix(const char* filename);

        /**
         * @brief Destroy the Sparse Matrix object
         */
        ~SparseMatrix();

        ///@}

        //* Getters *//
        /**
         * @name Getters
         */
         ///@{

         /**
          * @returns T The value at the specified row and column. Returns 0 if the
          * value is not found.
          *
          * Get the value at the specified row and column
          *
          * @note Users cannot update individual values in a IVSparse matrix.
          *
          * @warning This method is not efficient and should not be used in performance
          * critical code.
          */
        T coeff(uint32_t row, uint32_t col);

        /**
         * @returns true If the matrix is stored in column major format
         * @returns false If the matrix is stored in row major format
         *
         * See the storage order of the IVSparse matrix.
         */
        bool isColumnMajor() const;

        /**
         * @param vec The vector to get the values for
         * @returns A pointer to the values of a given vector in a VCSC Matrix
         */
        T* getValues(uint32_t vec) const;

        /**
         * @param vec The vector to get the counts for
         * @returns A pointer to the value counts of a given vector in a VCSC Matrix
         */
        indexT* getCounts(uint32_t vec) const;

        /**
         * @param vec The vector to get the indices for
         * @returns A pointer to the indices of a given vector in a VCSC Matrix
         */
        indexT* getIndices(uint32_t vec) const;

        /**
         * @param vec The vector to get the unique values for
         * @returns The number of unique values in a given vector in a VCSC Matrix
         */
        indexT getNumUniqueVals(uint32_t vec) const;

        /**
         * @param vec The vector to get the the number of indices for
         * @returns The number of indices (nonzeros) in a given vector in a VCSC
         * Matrix
         */
        indexT getNumIndices(uint32_t vec) const;

        /**
         * @param vec The vector to get a copy of
         * @returns Vector The vector copy returned
         *
         * Get a copy of a IVSparse vector from the IVSparse matrix such as the first
         * column.
         *
         * @note Can only get vectors in the storage order of the matrix.
         */
        typename IVSparse::SparseMatrix<T, indexT, 2, columnMajor>::Vector getVector(
            uint32_t vec);

        ///@}

        //* Calculations *//
        /**
         * @name Calculations
         */
         ///@{

         /**
          * @returns A vector of the sum of each vector along the outer dimension.
          */
        inline std::vector<T> outerSum();

        /**
         * @returns A vector of the sum of each vector along the inner dimension.
         */
        inline std::vector<T> innerSum();

        /**
         * @returns A vector of the maximum value in each column.
         */
        inline std::vector<T> maxColCoeff();

        /**
         * @returns A vector of the maximum value in each row.
         */
        inline std::vector<T> maxRowCoeff();

        /**
         * @returns A vector of the minimum value in each column.
         */
        inline std::vector<T> minColCoeff();

        /**
         * @returns A vector of the minimum value in each row.
         */
        inline std::vector<T> minRowCoeff();

        /**
         * @returns The trace of the matrix.
         *
         * @note Only works for square matrices.
         */
        inline T trace();

        /**
         * @returns The sum of all the values in the matrix.
         */
        inline T sum();

        /**
         * @returns The frobenius norm of the matrix.
         */
        inline double norm();

        /**
         * @returns Returns the length of the specified vector.
         */
        inline double vectorLength(uint32_t vec);

        ///@}

        //* Utility Methods *//
        /**
         * @name Utility Methods
         */
         ///@{

         /**
          * @param filename The filename of the matrix to write to
          *
          * This method writes the IVSparse matrix to a file in binary format.
          * This can then be read in later using the file constructor.
          * Currently .ivsparse is the perfered file extension.
          *
          * @note Useful to split a matrix up and then write each part separately.
          */
        void write(const char* filename);

        /**
         * Prints "IVSparse Matrix:" followed by the dense representation of the
         * matrix to the console.
         *
         * @note Useful for debugging but only goes up to 100 of either dimension.
         */
        void print();

        /**
         * @returns The current matrix as uncompressed to CSC format.
         */
        IVSparse::SparseMatrix<T, indexT, 1, columnMajor> toCSC();

        /**
         * @returns The current matrix as a IVCSC Matrix.
         */
        IVSparse::SparseMatrix<T, indexT, 3, columnMajor> toIVCSC();

        /**
         * @returns An Eigen Sparse Matrix constructed from the VCSC matrix data.
         */
        Eigen::SparseMatrix<T, columnMajor ? Eigen::ColMajor : Eigen::RowMajor>
            toEigen();

        ///@}

        //* Matrix Manipulation Methods *//
        /**
         * @name Matrix Manipulation Methods
         */
         ///@{

         /**
          * @returns A transposed version of the IVSparse matrix.
          *
          * @warning This method is not very efficient for VCSC and IVCSC matrices.
          */
        IVSparse::SparseMatrix<T, indexT, 2, columnMajor> transpose();

        /**
         * Transposes the matrix in place instead of returning a new matrix.
         *
         * @warning This method is not very efficient for VCSC and IVCSC matrices.
         */
        void inPlaceTranspose();

        /**
         * @param vec The vector to append to the matrix in the correct storage order.
         *
         * Appends a IVSparse vector to the current matrix in the storage order of the
         * matrix.
         */
        void append(typename SparseMatrix<T, indexT, 2, columnMajor>::Vector& vec);

        /**
         * @returns A vector of IVSparse vectors that represent a slice of the
         * IVSparse matrix.
         */
        std::vector<
            typename IVSparse::SparseMatrix<T, indexT, 2, columnMajor>::Vector>
            slice(uint32_t start, uint32_t end);

        ///@}

        //* Operator Overloads *//

        // Assignment Operator
        IVSparse::SparseMatrix<T, indexT, 2, columnMajor>& operator=(
            const IVSparse::SparseMatrix<T, indexT, 2, columnMajor>& other);

        // Equality Operator
        bool operator==(const SparseMatrix<T, indexT, 2, columnMajor>& other);

        // Inequality Operator
        bool operator!=(const SparseMatrix<T, indexT, 2, columnMajor>& other);

        // Coefficient Access Operator
        T operator()(uint32_t row, uint32_t col);

        // Vector Access Operator
        typename IVSparse::SparseMatrix<T, indexT, 2, columnMajor>::Vector operator[](
            uint32_t vec);

        // Scalar Multiplication
        IVSparse::SparseMatrix<T, indexT, 2, columnMajor> operator*(T scalar);

        // In Place Scalar Multiplication
        void operator*=(T scalar);

        // Matrix Vector Multiplication
        Eigen::Matrix<T, -1, 1> operator*(Eigen::Matrix<T, -1, 1>& vec);

        // Matrix Vector Multiplication 2 (with IVSparse Vector)
        Eigen::Matrix<T, -1, 1> operator*(
            typename SparseMatrix<T, indexT, 2, columnMajor>::Vector& vec);

        // Matrix Matrix Multiplication
        Eigen::Matrix<T, -1, -1> operator*(Eigen::Matrix<T, -1, -1>& mat);

    };  // End of VCSC Sparse Matrix Class

}  // namespace IVSparse
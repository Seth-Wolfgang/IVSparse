/**
 * @file IVCSCC_Methods.hpp
 * @author Skyler Ruiter and Seth Wolfgang
 * @brief Methods for IVCSC Sparse Matrices
 * @version 0.1
 * @date 2023-07-03
 */

#pragma once

namespace IVSparse {

    //* Getters *//

    // Gets the element stored at the given row and column
    template <typename T, typename indexT, uint8_t compressionLevel, bool columnMajor>
    T SparseMatrix<T, indexT, compressionLevel, columnMajor>::coeff(uint32_t row, uint32_t col) {

        #ifdef IVSPARSE_DEBUG
        // check that the row and column are valid
        assert(row < numRows && col < numCols && "Invalid row and column!");
        assert(row >= 0 && col >= 0 && "Invalid row and column!");
        #endif

        return (*this)(row, col);
    }

    // Check for Column Major
    template <typename T, typename indexT, uint8_t compressionLevel, bool columnMajor>
    bool SparseMatrix<T, indexT, compressionLevel, columnMajor>::isColumnMajor() const {
        return columnMajor;
    }

    // Returns a pointer to the given vector
    template <typename T, typename indexT, uint8_t compressionLevel, bool columnMajor>
    void* SparseMatrix<T, indexT, compressionLevel, columnMajor>::vectorPointer(uint32_t vec) {

        #ifdef IVSPARSE_DEBUG
        assert(vec < outerDim && vec >= 0 && "Invalid vector!");
        #endif

        return data[vec];
    }

    // Gets a IVSparse vector copy of the given vector
    template <typename T, typename indexT, uint8_t compressionLevel, bool columnMajor>
    typename SparseMatrix<T, indexT, compressionLevel, columnMajor>::Vector SparseMatrix<T, indexT, compressionLevel, columnMajor>::getVector(uint32_t vec) {
        #ifdef IVSPARSE_DEBUG
        assert(vec < outerDim && vec >= 0 && "Invalid vector!");
        #endif

        return (*this)[vec];
    }

    // Gets the byte size of a given vector
    template <typename T, typename indexT, uint8_t compressionLevel, bool columnMajor>
    size_t SparseMatrix<T, indexT, compressionLevel, columnMajor>::getVectorSize(uint32_t vec) const {
        #ifdef IVSPARSE_DEBUG
        assert(vec < outerDim && vec >= 0 && "Invalid vector!");
        #endif

        if (data[vec] == endPointers[vec]) {
            return 0;
        }
        return (char*)endPointers[vec] - (char*)data[vec];
    }

    //* Utility Methods *//

    // Writes the matrix to file
    template <typename T, typename indexT, uint8_t compressionLevel, bool columnMajor>
    void SparseMatrix<T, indexT, compressionLevel, columnMajor>::write(const char* filename) {

        // Open the file
        FILE* fp = fopen(filename, "wb+");

        // Write the metadata
        fwrite(metadata, 1, NUM_META_DATA * sizeof(uint32_t), fp);

        // write the size of each vector
        for (uint32_t i = 0; i < outerDim; i++) {
            uint64_t size = (uint8_t*)endPointers[i] - (uint8_t*)data[i];
            fwrite(&size, 1, sizeof(uint64_t), fp);
        }

        // write each vector
        for (uint32_t i = 0; i < outerDim; i++) {
            fwrite(data[i], 1, (char*)endPointers[i] - (char*)data[i], fp);
        }

        // close the file
        fclose(fp);
    }

    // Prints the matrix dense to console
    template <typename T, typename indexT, uint8_t compressionLevel, bool columnMajor>
    void SparseMatrix<T, indexT, compressionLevel, columnMajor>::print() {

        std::cout << std::endl;
        std::cout << "IVSparse Matrix" << std::endl;

        // if the matrix is less than 100 rows and columns print the whole thing
        if (numRows < 100 && numCols < 100) {
            // print the matrix
            for (uint32_t i = 0; i < numRows; i++) {
                for (uint32_t j = 0; j < numCols; j++) {
                    std::cout << coeff(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
        else if (numRows > 100 && numCols > 100) {
            // print the first 100 rows and columns
            for (uint32_t i = 0; i < 100; i++) {
                for (uint32_t j = 0; j < 100; j++) {
                    std::cout << coeff(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }

        std::cout << std::endl;
    }

    // Convert a IVCSC matrix to CSC
    template <typename T, typename indexT, uint8_t compressionLevel, bool columnMajor>
    IVSparse::SparseMatrix<T, indexT, 1, columnMajor> SparseMatrix<T, indexT, compressionLevel, columnMajor>::toCSC() {

        // create a new sparse matrix
        Eigen::SparseMatrix<T, columnMajor ? Eigen::ColMajor : Eigen::RowMajor> eigenMatrix(numRows, numCols);

        // iterate over the matrix
        for (uint32_t i = 0; i < outerDim; ++i) {
            for (typename SparseMatrix<T, indexT, compressionLevel>::InnerIterator it(
                *this, i);
                it; ++it) {
                // add the value to the matrix
                eigenMatrix.insert(it.row(), it.col()) = it.value();
            }
        }

        // finalize the matrix
        eigenMatrix.makeCompressed();

        // make a CSC matrix
        IVSparse::SparseMatrix<T, indexT, 1, columnMajor> CSCMatrix(eigenMatrix);

        // return the matrix
        return CSCMatrix;
    }

    // Convert a IVCSC matrix to a VCSC matrix
    template <typename T, typename indexT, uint8_t compressionLevel, bool columnMajor>
    IVSparse::SparseMatrix<T, indexT, 2, columnMajor> SparseMatrix<T, indexT, compressionLevel, columnMajor>::toVCSC() {
        // if already VCSC return a copy
        if (compressionLevel == 2) {
            return *this;
        }

        //* compress the data

        // make a pointer for the CSC pointers
        T* values = (T*)malloc(nnz * sizeof(T));
        indexT* indices = (indexT*)malloc(nnz * sizeof(indexT));
        indexT* colPtrs = (indexT*)malloc((outerDim + 1) * sizeof(indexT));

        colPtrs[0] = 0;

        // make an array of ordered maps to hold the data
        std::map<indexT, T> dict[outerDim];

        // iterate through the data using the iterator
        for (uint32_t i = 0; i < outerDim; ++i) {
            size_t count = 0;

            for (typename SparseMatrix<T, indexT, compressionLevel>::InnerIterator it(
                *this, i);
                it; ++it) {
                dict[i][it.getIndex()] = it.value();
                count++;
            }

            colPtrs[i + 1] = colPtrs[i] + count;
        }

        size_t count = 0;

        // loop through the dictionary and populate values and indices
        for (uint32_t i = 0; i < outerDim; ++i) {
            for (auto& pair : dict[i]) {
                values[count] = pair.second;
                indices[count] = pair.first;
                count++;
            }
        }

        // return a VCSC matrix from the CSC vectors
        IVSparse::SparseMatrix<T, indexT, 2, columnMajor> mat(
            values, indices, colPtrs, numRows, numCols, nnz);

        // free the CSC vectors
        free(values);
        free(indices);
        free(colPtrs);

        return mat;
    }

    // converts the ivsparse matrix to an eigen one and returns it
    template <typename T, typename indexT, uint8_t compressionLevel, bool columnMajor>
    Eigen::SparseMatrix<T, columnMajor ? Eigen::ColMajor : Eigen::RowMajor> SparseMatrix<T, indexT, compressionLevel, columnMajor>::toEigen() {
        #ifdef IVSPARSE_DEBUG
        // assert that the matrix is not empty
        assert(outerDim > 0 && "Cannot convert an empty matrix to an Eigen matrix!");
        #endif

        // create a new sparse matrix
        Eigen::SparseMatrix<T, columnMajor ? Eigen::ColMajor : Eigen::RowMajor> eigenMatrix(numRows, numCols);

        // iterate over the matrix
        for (uint32_t i = 0; i < outerDim; ++i) {
            // check if the vector is empty
            if (data[i] == nullptr) {
                continue;
            }

            for (typename SparseMatrix<T, indexT, compressionLevel>::InnerIterator it(
                *this, i);
                it; ++it) {
                // add the value to the matrix
                eigenMatrix.insert(it.row(), it.col()) = it.value();
            }
        }

        // finalize the matrix
        eigenMatrix.makeCompressed();

        // return the matrix
        return eigenMatrix;
    }

    //* Conversion/Transformation Methods *//

    // appends a vector to the back of the storage order of the matrix
    template <typename T, typename indexT, uint8_t compressionLevel, bool columnMajor>
    void SparseMatrix<T, indexT, compressionLevel, columnMajor>::append(
        typename IVSparse::SparseMatrix<T, indexT, compressionLevel, columnMajor>::Vector& vec) {

        #ifdef IVSPARSE_DEBUG
        assert(vec.getLength() == innerDim && "Vector must be the same size as the inner dimension!");
        #endif

        // check if the matrix is empty
        if (compSize == 0) [[unlikely]] {
            *this = IVSparse::SparseMatrix<T, indexT, compressionLevel, columnMajor>(vec);
            }
        else {
            // check if the vector is empty, if so change the implementation details
            if (vec.begin() == vec.end()) [[unlikely]] {
                if (columnMajor) {
                    numCols++;
                }
                else {
                    numRows++;
                }
                outerDim++;

                // update metadata
                metadata[2] = outerDim;

                // realloc the data to be one larger
                try {
                    data = (void**)realloc(data, outerDim * sizeof(void*));
                    endPointers = (void**)realloc(endPointers, outerDim * sizeof(void*));
                }
                catch (std::bad_alloc& e) {
                    throw std::bad_alloc();
                }

                // set the new vector to nullptr
                data[outerDim - 1] = nullptr;
                endPointers[outerDim - 1] = nullptr;

                calculateCompSize();

                return;
                }
            else {
                // check that the vector is the correct size
                if ((vec.getLength() != innerDim))
                    throw std::invalid_argument(
                        "The vector must be the same size as the outer dimension of the "
                        "matrix!");

                outerDim++;
                nnz += vec.nonZeros();
                if (columnMajor) {
                    numCols++;
                }
                else {
                    numRows++;
                }

                // update metadata
                metadata[2] = outerDim;
                metadata[3] = nnz;

                // realloc the data to be one larger
                try {
                    data = (void**)realloc(data, outerDim * sizeof(void*));
                    endPointers = (void**)realloc(endPointers, outerDim * sizeof(void*));
                }
                catch (std::bad_alloc& e) {
                    throw std::bad_alloc();
                }

                // malloc the new vector
                try {
                    data[outerDim - 1] = malloc(vec.byteSize());
                    endPointers[outerDim - 1] = (char*)data[outerDim - 1] + vec.byteSize();
                }
                catch (std::bad_alloc& e) {
                    throw std::bad_alloc();
                }

                // copy the vector into the new space
                memcpy(data[outerDim - 1], vec.begin(), vec.byteSize());

                calculateCompSize();
            }
        }

    }  // end append

    // tranposes the ivsparse matrix
    template <typename T, typename indexT, uint8_t compressionLevel, bool columnMajor>
    IVSparse::SparseMatrix<T, indexT, compressionLevel, columnMajor> SparseMatrix<T, indexT, compressionLevel, columnMajor>::transpose() {

        // make a data structure to store the tranpose
        // std::unordered_map<T, std::vector<indexT>> mapsT[innerDim];
        std::unordered_map<T, std::vector<indexT>>* mapsT = (std::unordered_map<T, std::vector<indexT>>*)malloc(innerDim * sizeof(std::unordered_map<T, std::vector<indexT>>));
        // std::vector<std::unordered_map<T, std::vector<indexT>>> mapsT;
        // mapsT.reserve(innerDim);
        // mapsT.resize(innerDim);
        // populate the transpose data structure
        for (uint32_t i = 0; i < outerDim; ++i) {
            for (typename SparseMatrix<T, indexT, compressionLevel>::InnerIterator it(*this, i); it; ++it) {
                // add the value to the map
                if constexpr (columnMajor) {
                    mapsT[it.row()][it.value()].push_back(it.col());
                }
                else {
                    mapsT[it.col()][it.value()].push_back(it.row());
                }
            }
        }


        for(int i = 0; i < innerDim; ++i) {
            for (auto& col : mapsT[i]) {
                // find the max value in the vector
                size_t max = col.second[0];

                // delta encode the vector
                for (uint32_t i = col.second.size() - 1; i > 0; --i) {
                    col.second[i] -= col.second[i - 1];
                    if ((size_t)col.second[i] > max) max = col.second[i];
                }

                max = byteWidth(max);
                // append max to the vector
                col.second.push_back(max);
            }
        }

        // create a new matrix passing in transposedMap
        IVSparse::SparseMatrix<T, indexT, compressionLevel, columnMajor> temp(mapsT, numRows, numCols);

        // return the new matrix
        return temp;
    }

    // Transpose In Place Method
    template <typename T, typename indexT, uint8_t compressionLevel, bool columnMajor>
    void SparseMatrix<T, indexT, compressionLevel, columnMajor>::inPlaceTranspose() {
        // make a data structure to store the tranpose
        // std::unordered_map<T, std::vector<indexT>> mapsT[innerDim];
        std::vector<std::unordered_map<T, std::vector<indexT>>> mapsT;
        // mapsT.reserve(innerDim);
        mapsT.resize(innerDim);

        // populate the transpose data structure
        for (uint32_t i = 0; i < outerDim; ++i) {
            for (typename SparseMatrix<T, indexT, compressionLevel>::InnerIterator it(*this, i); it; ++it) {
                // add the value to the map
                if constexpr (columnMajor) {
                    mapsT[it.row()][it.value()].push_back(it.col());
                }
                else {
                    mapsT[it.col()][it.value()].push_back(it.row());
                }
            }
        }

        for(int i = 0; i < innerDim; ++i) {
            for (auto& col : mapsT[i]) {
                // find the max value in the vector
                size_t max = col.second[0];

                // delta encode the vector
                for (uint32_t i = col.second.size() - 1; i > 0; --i) {
                    col.second[i] -= col.second[i - 1];
                    if ((size_t)col.second[i] > max) max = col.second[i];
                }

                max = byteWidth(max);
                // append max to the vector
                col.second.push_back(max);
            }
        }

        *this = IVSparse::SparseMatrix<T, indexT, compressionLevel, columnMajor>(mapsT, numRows, numCols);
    }

    // slice method that returns a vector of IVSparse vectors
    template <typename T, typename indexT, uint8_t compressionLevel, bool columnMajor>
    std::vector<typename IVSparse::SparseMatrix<T, indexT, compressionLevel, columnMajor>::Vector>
        SparseMatrix<T, indexT, compressionLevel, columnMajor>::slice(uint32_t start, uint32_t end) {

        #ifdef IVSPARSE_DEBUG
        assert(start < outerDim && end <= outerDim && start < end &&
               "Invalid start and end values!");
        #endif

        // make a vector of IVSparse vectors
        std::vector<typename IVSparse::SparseMatrix<T, indexT, compressionLevel,
            columnMajor>::Vector>
            vecs(end - start);

        // grab the vectors and add them to vecs
        for (uint32_t i = start; i < end; ++i) {
            // make a temp vector
            IVSparse::SparseMatrix<T, indexT, compressionLevel, columnMajor>::Vector
                temp(*this, i);

            // add the vector to vecs
            vecs[i - start] = temp;
        }

        // return the vector
        return vecs;
    }

}  // end namespace IVSparse
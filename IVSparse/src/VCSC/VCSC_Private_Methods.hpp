/**
 * @file VCSC_Private_Methods.hpp
 * @author Skyler Ruiter and Seth Wolfgang
 * @brief Private Methods for VCSC Sparse Matrices
 * @version 0.1
 * @date 2023-07-03
 */

#pragma once

namespace IVSparse {

    // Encodes the value type of the matrix in a uint32_t
    template <typename T, typename indexT, bool columnMajor>
    void SparseMatrix<T, indexT, 2, columnMajor>::encodeValueType() {
        uint8_t byte0 = sizeof(T);
        uint8_t byte1 = std::is_floating_point<T>::value ? 1 : 0;
        uint8_t byte2 = std::is_signed_v<T> ? 1 : 0;
        uint8_t byte3 = columnMajor ? 1 : 0;

        val_t = (byte3 << 24) | (byte2 << 16) | (byte1 << 8) | byte0;
    }

    // Checks if the value type is correct for the matrix
    template <typename T, typename indexT, bool columnMajor>
    void SparseMatrix<T, indexT, 2, columnMajor>::checkValueType() {
        uint8_t byte0 = val_t & 0xFF;
        uint8_t byte1 = (val_t >> 8) & 0xFF;
        uint8_t byte2 = (val_t >> 16) & 0xFF;
        uint8_t byte3 = (val_t >> 24) & 0xFF;
        assert(byte0 == sizeof(T) && "Value type size does not match");
        assert(byte1 == std::is_floating_point_v<T> &&
               "Value type is not floating point");
        assert(byte2 == std::is_signed_v<T> && "Value type is not signed");
        assert(byte3 == columnMajor && "Major direction does not match");
    }

    // performs some simple user checks on the matrices metadata
    template <typename T, typename indexT, bool columnMajor>
    void SparseMatrix<T, indexT, 2, columnMajor>::userChecks() {
        assert((innerDim > 1 || outerDim > 1 || nnz > 1) &&
               "The matrix must have at least one row, column, and nonzero value");
        assert(std::is_floating_point<indexT>::value == false &&
               "The index type must be a non-floating point type");
        assert((std::is_arithmetic<T>::value && std::is_arithmetic<indexT>::value) &&
               "The value and index types must be numeric types");
        assert((std::is_same<indexT, bool>::value == false) &&
               "The index type must not be bool");
        assert((innerDim < std::numeric_limits<indexT>::max() &&
                outerDim < std::numeric_limits<indexT>::max()) &&
               "The number of rows and columns must be less than the maximum value "
               "of the index type");
        checkValueType();
    }

    // Calculates the current byte size of the matrix in memory
    template <typename T, typename indexT, bool columnMajor>
    void SparseMatrix<T, indexT, 2, columnMajor>::calculateCompSize() {
        // set compSize to zero
        compSize = 0;

        compSize += (sizeof(indexT) * outerDim);  // valueSizes 4


        for (uint32_t i = 0; i < outerDim; i++) {

            compSize += (sizeof(T) * valueSizes[i]);       // values 8 -> 8 per value
            compSize += (sizeof(indexT) * valueSizes[i]);  // counts 4 -> 4 per value
            compSize += (sizeof(indexT) * indexSizes[i]);  // indices 4 -> 4 per index
        }
    }

    // Compression Algorithm for going from CSC to VCSC
    template <typename T, typename indexT, bool columnMajor>
    template <typename T2, typename indexT2>
    void SparseMatrix<T, indexT, 2, columnMajor>::compressCSC(T2* vals, indexT2* innerIndices, indexT2* outerPointers) {
        // ---- Stage 1: Setup the Matrix ---- //

        // set the value and index types of the matrix
        encodeValueType();
        index_t = sizeof(indexT);

        // allocate space for metadata
        metadata = new uint32_t[NUM_META_DATA];
        metadata[0] = 2;
        metadata[1] = innerDim;
        metadata[2] = outerDim;
        metadata[3] = nnz;
        metadata[4] = val_t;
        metadata[5] = index_t;

        // run the user checks on the metadata
        #ifdef IVSPARSE_DEBUG
        userChecks();
        #endif

        // allocate space for the 2D Run lenngth encoded CSC matrix
        try {
            values = (T**)malloc(sizeof(T*) * outerDim);
            counts = (indexT**)malloc(sizeof(indexT*) * outerDim);
            indices = (indexT**)malloc(sizeof(indexT*) * outerDim);

            valueSizes = (indexT*)malloc(sizeof(indexT) * outerDim);
            indexSizes = (indexT*)malloc(sizeof(indexT) * outerDim);
        }
        catch (std::bad_alloc& e) {
            std::cerr << "Error: Could not allocate memory for the matrix" << std::endl;
            exit(1);
        }

        // ---- Stage 2: Construct the Dictionary For Each Column ---- //

        // Loop through each column and construct a middle data structre for the matrix
        #ifdef IVSPARSE_HAS_OPENMP
        #pragma omp parallel for
        #endif
        for (uint32_t i = 0; i < outerDim; i++) {
            
            // create the data structure to temporarily hold the data
            std::map<T2, std::vector<indexT2>> dict;  // Key = value, Value = vector of indices

            // check if the current column is empty
            if (outerPointers[i] == outerPointers[i + 1]) {
                valueSizes[i] = 0;
                indexSizes[i] = 0;

                values[i] = nullptr;
                counts[i] = nullptr;
                indices[i] = nullptr;
                continue;
            }

            // create a variable to hold the size of the column
            size_t numIndices = 0;

            // loop through each value in the column and add it to dict
            for (indexT2 j = outerPointers[i]; j < outerPointers[i + 1]; j++) {
                // check if the value is already in the dictionary or not
                if (dict.find(vals[j]) != dict.end()) {
                    // add the index
                    dict[vals[j]].push_back(innerIndices[j]);

                    numIndices++;
                }
                else {
                    // if value not already in the dictionary add it

                    // create a new vector for the indices
                    dict[vals[j]] = std::vector<indexT2>{ innerIndices[j] };

                    numIndices++;
                }

            }  // end value loop

            // ---- Stage 3: Allocate Size of Column Data ---- //

            try {
                values[i] = (T*)malloc(sizeof(T) * dict.size());
                counts[i] = (indexT*)malloc(sizeof(indexT) * dict.size());
                indices[i] = (indexT*)malloc(sizeof(indexT) * numIndices);
            }
            catch (std::bad_alloc& e) {
                std::cerr << "Error: Could not allocate memory for the matrix"
                    << std::endl;
                exit(1);
            }

            // set the size of the column
            valueSizes[i] = dict.size();
            indexSizes[i] = numIndices;
            size_t performanceVecSize = 0;
            size_t indexSize = 0;

            // ---- Stage 4: Populate the Column Data ---- //

            for (auto& pair : dict) {
                values[i][performanceVecSize] = pair.first;
                counts[i][performanceVecSize] = pair.second.size();

                // memcpy the indices into the indices array and increment the indexSize
                memcpy(&indices[i][indexSize], &pair.second[0], sizeof(indexT) * pair.second.size());
                indexSize += pair.second.size();


                performanceVecSize++;
            }

        }  // end column loop

        calculateCompSize();

    }  // end compressCSC

}  // end namespace IVSparse
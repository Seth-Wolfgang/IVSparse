#include <Eigen/Sparse>


class rng {
private:
    uint64_t state;

public:
    rng(uint64_t state) : state(state) {}

    void advance_state() {
        state ^= state << 19;
        state ^= state >> 7;
        state ^= state << 36;
    }

    uint64_t operator*() const {
        return state;
    }

    uint64_t rand() {
        uint64_t x = state ^ (state << 38);
        x ^= x >> 13;
        x ^= x << 23;

        return x;
    }

    uint64_t rand(uint64_t i) {
        // advance i
        i ^= i << 19;
        i ^= i >> 7;
        i ^= i << 36;

        // add i to state
        uint64_t x = state + i;

        // advance state
        x ^= x << 38;
        x ^= x >> 13;
        x ^= x << 23;

        return x;
    }

    uint64_t rand(uint64_t i, uint64_t j) {
        uint64_t x = rand(i);

        // advance j
        j ^= j >> 7;
        j ^= j << 23;
        j ^= j >> 8;

        // add j to state
        x += j;

        // advance state
        x ^= x >> 7;
        x ^= x << 53;
        x ^= x >> 4;

        return x;
    }

    template <typename T>
    T sample(T max_value) {
        return rand() % max_value;
    }

    template <typename T>
    T sample(uint64_t i, T max_value) {
        return rand(i) % max_value;
    }

    template <typename T>
    T sample(uint64_t i, uint64_t j, T max_value) {

        if constexpr (std::is_floating_point<T>::value)
            return fmod((double)rand(i, j), max_value);
        else
            return rand(i, j) % max_value;
    }

    template <typename T>
    bool draw(T probability) {
        return sample(probability) == 0;
    }

    template <typename T>
    bool draw(uint64_t i, T probability) {
        return sample(i, probability) == 0;
    }

    template <typename T>
    bool draw(uint64_t i, uint64_t j, T probability) {
        if constexpr (std::is_floating_point<T>::value)
            return sample(i, j, probability) <= 0.003;
        else
            return sample(i, j, probability) == 0;
    }

    template <typename T>
    double uniform() {
        T x = (T)rand() / UINT64_MAX;
        return x - std::floor(x);
    }

    template <typename T>
    double uniform(uint64_t i) {
        T x = (T)rand(i) / UINT64_MAX;
        return x - std::floor(x);
    }

    template <typename T>
    double uniform(uint64_t i, uint64_t j) {
        T x = (T)rand(i, j) / UINT64_MAX;
        return x - std::floor(x);
    }
};

template <typename T>
Eigen::SparseMatrix<T> generateMatrix(int numRows, int numCols, int sparsity, uint64_t seed, uint64_t maxValue) {
    // generate a random sparse matrix
    rng randMatrixGen = rng(seed);

    Eigen::SparseMatrix<T> myMatrix(numRows, numCols);
    myMatrix.reserve(Eigen::VectorXi::Constant(numCols, numRows));

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            if (randMatrixGen.draw<int>(i, j, sparsity)) {
                myMatrix.insert(i, j) = rand() % maxValue + 1;
            }
        }
    }

    myMatrix.makeCompressed();
    return myMatrix;
}

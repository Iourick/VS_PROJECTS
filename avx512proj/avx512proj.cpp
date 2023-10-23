#include <iostream>
#include <immintrin.h>  // Include AVX2 intrinsics

int main() {
    const int N = 16; // Number of elements in the arrays

    // Initialize two arrays with random data
    float A[N], B[N], C[N];
    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i * 2);
    }

    // Perform vectorized addition using AVX2 intrinsics
    for (int i = 0; i < N; i += 8) {
        __m256 a = _mm256_loadu_ps(&A[i]); // Load 8 floats from A
        __m256 b = _mm256_loadu_ps(&B[i]); // Load 8 floats from B
        __m256 result = _mm256_add_ps(a, b); // Perform addition

        _mm256_storeu_ps(&C[i], result); // Store the result in C
    }

    // Print the result
    std::cout << "Result: ";
    for (int i = 0; i < N; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>
#include <iostream>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
#define RAND_VEC3 Vec3(curand_uniform(rand_state), curand_uniform(rand_state), curand_uniform(rand_state))

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<uint32_t>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        std::exit(99);
    }
}

// Constants
constexpr float INF = std::numeric_limits<float>::infinity();
constexpr float PI = 3.1415926535897932385f;

// Common Headers
#include "Ray.cuh"
#include "Vec3.cuh"
#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>
#include <iostream>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
#define RAND_VEC3 Vec3(curand_uniform(rand_state), curand_uniform(rand_state), curand_uniform(rand_state))
#define RAND (curand_uniform(&local_rand_state))
#define FLT_MAX 3.402823466e+38F
#define INF std::numeric_limits<float>::infinity()
#define PI 3.1415926535897932385f

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

__device__ inline float degreesToRadians(float degrees)
{
    return degrees * PI / 180.0f;
}

__device__ inline float clamp(float x, float min, float max)
{
    if (x < min)
    {
        return min;
    }
    if (x > max)
    {
        return max;
    }

    return x;
}

// Common Headers
#include "Ray.cuh"
#include "Vec3.cuh"
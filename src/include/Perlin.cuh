#pragma once

#include "Util.cuh"

class Perlin
{
    static constexpr int point_count = 256;
    int *perm_x;
    int *perm_y;
    int *perm_z;
    Vec3 *rand_vec;

    __device__ static int *perlinGeneratePerm(curandState *rand_state);
    __device__ static void permute(int *p, int n, curandState *rand_state);

public:
    __device__ Perlin(curandState *rand_state);
    __device__ ~Perlin();
    __device__ float noise(const Point3D &p) const;
    __device__ float trilinerInterp(Vec3 c[2][2][2], float u, float v, float w) const;
    __device__ float turb(const Point3D &p, int depth = 7) const;
};

__device__ int *Perlin::perlinGeneratePerm(curandState *rand_state)
{
    int *p = new int[point_count];

    for (int i = 0; i < Perlin::point_count; ++i)
    {
        p[i] = i;
    }

    permute(p, Perlin::point_count, rand_state);

    return p;
}

__device__ void Perlin::permute(int *p, int n, curandState *rand_state)
{
    for (int i = n - 1; i > 0; --i)
    {
        int target = curand(rand_state) % (i + 1);
        int tmp = p[i];
        p[i] = p[target];
        p[target] = tmp;
    }
}

__device__ Perlin::Perlin(curandState *rand_state) : perm_x(perlinGeneratePerm(rand_state)), perm_y(perlinGeneratePerm(rand_state)), perm_z(perlinGeneratePerm(rand_state))
{
    rand_vec = new Vec3[point_count];
    for (int i = 0; i < point_count; ++i)
    {
        rand_vec[i] = randomUnitVector(rand_state);
    }
}

__device__ Perlin::~Perlin()
{
    delete[] rand_vec;
    delete[] perm_x;
    delete[] perm_y;
    delete[] perm_z;
}

__device__ float Perlin::noise(const Point3D &p) const
{
    float u = p.x - floor(p.x);
    float v = p.y - floor(p.y);
    float w = p.z - floor(p.z);

    int i = static_cast<int>(floor(p.x));
    int j = static_cast<int>(floor(p.y));
    int k = static_cast<int>(floor(p.z));
    Vec3 c[2][2][2];

    for (int di = 0; di < 2; ++di)
    {
        for (int dj = 0; dj < 2; ++dj)
        {
            for (int dk = 0; dk < 2; ++dk)
            {
                c[di][dj][dk] = rand_vec[perm_x[(i + di) & 255] ^ perm_y[(j + dj) & 255] ^ perm_z[(k + dk) & 255]];
            }
        }
    }

    return trilinerInterp(c, u, v, w);
}

__device__ float Perlin::trilinerInterp(Vec3 c[2][2][2], float u, float v, float w) const
{
    float uu = u * u * (3 - 2 * u);
    float vv = v * v * (3 - 2 * v);
    float ww = w * w * (3 - 2 * w);
    float accum = 0.0f;

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                Vec3 weight_v(u - i, v - j, w - k);
                accum += (i * uu + (1 - i) * (1 - uu)) * (j * vv + (1 - j) * (1 - vv)) * (k * ww + (1 - k) * (1 - ww)) * dotProd(c[i][j][k], weight_v);
            }
        }
    }

    return accum;
}

__device__ float Perlin::turb(const Point3D &p, int depth) const
{
    float accum = 0.0f;
    Point3D temp_p = p;
    float weight = 1.0f;

    for (int i = 0; i < depth; ++i)
    {
        accum += weight * noise(temp_p);
        weight *= 0.5f;
        temp_p *= 2.0f;
    }

    return fabs(accum);
}
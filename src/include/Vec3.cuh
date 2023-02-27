#pragma once

#include "Util.cuh"

#include <cmath>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <curand_kernel.h>

struct Vec3
{
    float x = 0.0f, y = 0.0f, z = 0.0f;

    // constructors
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3() = default;

    // inversion operator
    __host__ __device__ Vec3 operator-() const
    {
        return Vec3(-x, -y, -z);
    }
    // access operator
    __host__ __device__ float operator[](int i) const
    {
        assert(i >= 0 && i <= 2);
        if (i == 0)
        {
            return x;
        }
        else if (i == 1)
        {
            return y;
        }
        else
        {
            return z;
        }
    }
    __host__ __device__ float &operator[](int i)
    {
        assert(i >= 0 && i <= 2);
        if (i == 0)
        {
            return x;
        }
        else if (i == 1)
        {
            return y;
        }
        else
        {
            return z;
        }
    }

    // addition-assignment operator
    __host__ __device__ Vec3 &operator+=(const Vec3 &other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    // multiplication-assignment operator
    __host__ __device__ Vec3 &operator*=(float t)
    {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    __host__ __device__ Vec3 &operator*=(Vec3 other)
    {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }

    // division-assignment operator
    __host__ __device__ Vec3 &operator/=(float t)
    {
        return *this *= 1.0f / t;
    }

    __host__ __device__ float squaredLength() const
    {
        return x * x + y * y + z * z;
    }

    // euclidean distance from the origin
    __host__ __device__ float length() const
    {
        return sqrt(squaredLength());
    }

    // return true if the vector is close to 0 in all dimensions
    __host__ __device__ inline bool nearZero() const
    {
        constexpr float epi = 1e-10;
        return (abs(x) < epi) && (abs(y) < epi) && (abs(z) < epi);
    }
};

// type aliases for Vec3
using Point3D = Vec3; // 3D point
using Color = Vec3;   // RGB color

//
// Vec3 utility functions
//

inline std::ostream &operator<<(std::ostream &out, const Vec3 &v)
{
    return out << v.x << " " << v.y << " " << v.z;
}

__host__ __device__ inline Vec3 operator+(const Vec3 &u, const Vec3 &v)
{
    return Vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

__host__ __device__ inline Vec3 operator-(const Vec3 &u, const Vec3 &v)
{
    return u + (-v);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &u, const Vec3 &v)
{
    return Vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3 &v)
{
    return Vec3(v.x * t, v.y * t, v.z * t);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v, float t)
{
    return t * v;
}

__host__ __device__ inline Vec3 operator/(const Vec3 &v, float t)
{
    return (1.0f / t) * v;
}

__host__ __device__ inline float dotProd(const Vec3 &u, const Vec3 &v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

__host__ __device__ inline Vec3 crossProd(const Vec3 &u, const Vec3 &v)
{
    return Vec3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}

__host__ __device__ inline Vec3 unitVec(const Vec3 &v)
{
    return v / v.length();
}

// sample a random point that resides inside the unit sphere
__device__ inline Point3D randomInUnitSphere(curandState *rand_state)
{
    Point3D ret;
    do
    {
        ret = 2.0f * RAND_VEC3 - Point3D(1.0f, 1.0f, 1.0f);
    } while (ret.squaredLength() >= 1.0f);

    return ret;
}

__device__ inline Vec3 randomUnitVector(curandState *rand_state)
{
    return unitVec(randomInUnitSphere(rand_state));
}

__device__ inline Point3D randomInHemisphere(const Vec3 &normal, curandState *rand_state)
{
    Point3D in_unit_sphere = randomInUnitSphere(rand_state);
    if (dotProd(in_unit_sphere, normal) > 0.0f) // in the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

__device__ inline Vec3 randomInUnitDisk(curandState *rand_state)
{
    Vec3 ret;
    do
    {
        ret = 2.0f * Vec3(curand_uniform(rand_state), curand_uniform(rand_state), 0.0f) - Vec3(1.0f, 1.0f, 0.0f);
    } while (dotProd(ret, ret) >= 1.0f);
    return ret;
}

__host__ __device__ inline Vec3 reflect(const Vec3 &v, const Vec3 &n)
{
    return v - 2.0f * dotProd(v, n) * n;
}

__host__ __device__ inline Vec3 refract(const Vec3 &uv, const Vec3 &n, float eta_ratio)
{
    float cos_theta = min(dotProd(-uv, n), 1.0f);
    Vec3 r_out_perp = eta_ratio * (uv + cos_theta * n);
    Vec3 r_out_par = -sqrt(abs(1.0f - r_out_perp.squaredLength())) * n;
    return r_out_perp + r_out_par;
}
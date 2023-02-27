#pragma once

#include "Vec3.cuh"

struct Ray
{
    Point3D origin;
    Vec3 direction;

    __device__ Ray() = default;
    __device__ Ray(const Point3D &origin, const Vec3 &direction) : origin(origin), direction(direction) {}

    __device__ Point3D at(float t) const
    {
        return origin + t * direction;
    }
};
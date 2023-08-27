#pragma once

#include "Vec3.cuh"

struct Ray
{
    Point3D origin;
    Vec3 direction;

    __device__ Ray() = default;
    __device__ Ray(const Point3D &origin, const Vec3 &direction) : origin(origin), direction(direction) {}


    /**
     * @brief Returns the point at a given distance along the ray
     * @param dist Distance along the ray
     * @return Point at a given distance along the ray
    */
    __device__ Point3D at(float dist) const
    {
        return origin + dist * direction;
    }
};
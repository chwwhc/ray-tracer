#pragma once

#include "Vec3.hh"

struct Ray
{
    Point3D origin;
    Vec3 direction;

    Ray() = default;
    Ray(const Point3D &origin, const Vec3 &direction) : origin(origin), direction(direction) {}

    Point3D at(double t) const
    {
        return origin + t * direction;
    }
};
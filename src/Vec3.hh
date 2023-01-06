#pragma once

#include <cmath>
#include <cassert>
#include <iostream>
#include <stdexcept>

struct Vec3
{
    double x = 0, y = 0, z = 0;

    // constructors
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    Vec3() = default;

    // inversion operator
    Vec3 operator-() const
    {
        return Vec3(-x, -y, -z);
    }
    // access operator
    double operator[](int i) const
    {
        switch (i)
        {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        default:
            throw std::invalid_argument("invalid index for Vec3(can only be within [0, 2])");
        }
    }
    double &operator[](int i)
    {
        switch (i)
        {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        default:
            throw std::invalid_argument("invalid index for Vec3(can only be within [0, 2])");
        }
    }

    // addition-assignment operator
    Vec3 &operator+=(const Vec3 &other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    // multiplication-assignment operator
    Vec3 &operator*=(double t)
    {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    // division-assignment operator
    Vec3 &operator/=(double t)
    {
        return *this *= 1 / t;
    }

    double squaredLength() const
    {
        return x * x + y * y + z * z;
    }

    // euclidean distance from the origin
    double length() const
    {
        return std::sqrt(squaredLength());
    }

    inline static Vec3 random()
    {
        return Vec3(randomDouble(), randomDouble(), randomDouble());
    }

    inline static Vec3 random(double min, double max)
    {
        return Vec3(randomDouble(min, max), randomDouble(min, max), randomDouble(min, max));
    }

    // return true if the vector is close to 0 in all dimensions
    inline bool nearZero() const
    {
        constexpr double epi = 1e-10;
        return (std::fabs(x) < epi) && (std::fabs(y) < epi) && (std::fabs(z) < epi);
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

inline Vec3 operator+(const Vec3 &u, const Vec3 &v)
{
    return {u.x + v.x, u.y + v.y, u.z + v.z};
}

inline Vec3 operator-(const Vec3 &u, const Vec3 &v)
{
    return u + (-v);
}

inline Vec3 operator*(const Vec3 &u, const Vec3 &v)
{
    return {u.x * v.x, u.y * v.y, u.z * v.z};
}

inline Vec3 operator*(double t, const Vec3 &v)
{
    return {v.x * t, v.y * t, v.z * t};
}

inline Vec3 operator*(const Vec3 &v, double t)
{
    return t * v;
}

inline Vec3 operator/(const Vec3 &v, double t)
{
    return (1 / t) * v;
}

inline double dotProd(const Vec3 &u, const Vec3 &v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

inline Vec3 crossProd(const Vec3 &u, const Vec3 &v)
{
    return {u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
}

inline Vec3 unitVec(const Vec3 &v)
{
    return v / v.length();
}

// sample a random point that resides inside the unit sphere
inline Vec3 randomInUnitSphere()
{
    while (true)
    {
        Vec3 p = Vec3::random(-1, 1);
        if (p.squaredLength() >= 1)
            continue;
        return p;
    }
}

inline Vec3 randomUnitVector()
{
    return unitVec(randomInUnitSphere());
}

inline Vec3 randomInHemisphere(const Vec3 &normal)
{
    Vec3 in_unit_sphere = randomInUnitSphere();
    if (dotProd(in_unit_sphere, normal) > 0.0) // in the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

inline Vec3 reflect(const Vec3 &v, const Vec3 &n)
{
    return v - 2 * dotProd(v, n) * n;
}

inline Vec3 refract(const Vec3 &uv, const Vec3 &n, double eta_ratio)
{
    double cos_theta = std::fmin(dotProd(-uv, n), 1.0);
    Vec3 r_out_perp = eta_ratio * (uv + cos_theta * n);
    Vec3 r_out_par = -std::sqrt(std::fabs(1.0 - r_out_perp.squaredLength())) * n;
    return r_out_perp + r_out_par;
}

inline Vec3 randomInUnitDisk()
{
    while (true)
    {
        Vec3 p(randomDouble(-1, 1), randomDouble(-1, 1), 0);
        if (p.squaredLength() >= 1)
            continue;
        return p;
    }
}
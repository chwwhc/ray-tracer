#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>

// Usings
using std::make_shared;
using std::shared_ptr;

// Constants
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double PI = 3.1415926535897932385;

// Utility Functions
inline double degreeToRadians(double degrees)
{
    return degrees * PI / 180.0;
}

inline double clamp(double x, double min, double max)
{
    return x < min ? min : (x > max ? max : x);
}

// Random number generator(0 <= r < 1)
inline double randomDouble()
{
    return std::rand() / (RAND_MAX + 1.0);
}

inline double randomDouble(double min, double max)
{
    return min + (max - min) * randomDouble();
}

// Common Headers
#include "Ray.hh"
#include "Vec3.hh"
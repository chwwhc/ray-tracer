#pragma once

#include "Util.hh"

class Camera
{
    Point3D origin;
    Point3D lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;
    double lens_radius;

public:
    Camera(Point3D lookfrom, Point3D lookat, Vec3 vup, double vert_fov, double aspect_ratio, double aperture, double focus_dist)
    {
        double theta = degreeToRadians(vert_fov);
        double h = std::tan(theta / 2);
        double viewport_height = 2.0 * h;
        double viewport_width = aspect_ratio * viewport_height;

        w = unitVec(lookfrom - lookat);
        u = unitVec(crossProd(vup, w));
        v = crossProd(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;

        lens_radius = aperture / 2;
    }

    Ray getRay(double s, double t) const
    {
        Vec3 rd = lens_radius * randomInUnitDisk();
        Vec3 offset = u * rd.x + v * rd.y;

        return {origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset};
    }
};
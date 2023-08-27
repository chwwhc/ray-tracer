#pragma once

#include "Util.cuh"

class Camera
{
    Point3D origin;
    Point3D lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;
    float lens_radius;

public:
    __device__ Camera(float vfov, float aspect_ratio, float aperture, float focus_dist, Point3D lookfrom, Point3D lookat, Vec3 vup)
    {
        const float theta = vfov * PI / 180.0f;
        const float half_height = tan(theta / 2.0f);
        const float half_width = aspect_ratio * half_height;

        this->w = unitVec(lookfrom - lookat);
        this->u = unitVec(crossProd(vup, w));
        this->v = crossProd(w, u);
        this->origin = lookfrom;
        this->horizontal = 2.0f * half_width * focus_dist * u;
        this->vertical = 2.0f * half_height * focus_dist * v;
        this->lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        this->lens_radius = aperture / 2.0f;
    }

    /**
     * @brief Get the Ray object
     * @param s horizontal offset
     * @param t vertical offset
     * @param rand_state random state
     * @return Ray
     */
    __device__ Ray getRay(float s, float t, curandState *rand_state) const
    {
        Vec3 rd = lens_radius * randomInUnitDisk(rand_state);
        Vec3 offset = u * rd.x + v * rd.y;

        return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }
};
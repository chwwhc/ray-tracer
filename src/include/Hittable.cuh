#pragma once

#include "Util.cuh"

struct Material;

/**
 * Record the state of a ray(whether it hits something or not etc)
 */
struct HitRecord
{
    Point3D hit_point;
    Vec3 normal;
    Material* mat_ptr;
    float t;
    bool is_front_face;

    __device__ inline void setFaceNormal(const Ray &r, const Vec3 &outward_normal)
    {
        is_front_face = dotProd(r.direction, outward_normal) < 0.0f;
        normal = is_front_face ? outward_normal : -outward_normal;
    }
};

// Hittable interface
struct Hittable
{
    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const = 0;
};
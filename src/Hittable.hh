#pragma once

#include "Util.hh"

struct Material;

/**
 * Record the state of a ray(whether it hits something or not etc)
 */
struct HitRecord
{
    Point3D p;
    Vec3 normal;
    shared_ptr<Material> mat_ptr;
    double t;
    bool front_face;

    inline void setFaceNormal(const Ray &r, const Vec3 &outward_normal)
    {
        front_face = dotProd(r.direction, outward_normal) < 0.0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

// Hittable interface
struct Hittable
{
    virtual bool hit(const Ray &r, double t_min, double t_max, HitRecord &rec) const = 0;
    virtual ~Hittable() = default;
};
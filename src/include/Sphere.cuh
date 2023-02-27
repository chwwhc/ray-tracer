#pragma once

#include "Hittable.cuh"
#include "Vec3.cuh"

struct Sphere : Hittable
{
    Point3D center;
    float radius;
    Material *mat_ptr;

    __device__ Sphere() = default;
    __device__ Sphere(Point3D center, float radius, Material *mat_ptr) : center(center), radius(radius), mat_ptr(mat_ptr) {}

    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const override;
};

__device__ bool Sphere::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const
{
    Vec3 oc = r.origin - center;
    float a = r.direction.squaredLength();
    float half_b = dotProd(oc, r.direction);
    float c = oc.squaredLength() - radius * radius;

    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f)
        return false;
    float sqrtd = sqrt(discriminant);

    // find the nearest root that lies in the acceptable range
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max)
    {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max)
            return false;
    }

    rec.t = root;
    rec.hit_point = r.at(rec.t);
    rec.normal = (rec.hit_point - center) / radius;
    Vec3 outward_normal = (rec.hit_point - center) / radius;
    rec.setFaceNormal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

// check if a ray hits the sphere or not
// using the simplified quadratic equation
__device__ float hitSphere(const Point3D &center, float radius, const Ray &r)
{
    Vec3 oc = r.origin - center;
    float a = r.direction.squaredLength();
    float half_b = dotProd(oc, r.direction);
    float c = oc.squaredLength() - radius * radius;
    float discriminant = half_b * half_b - a * c;

    return discriminant < 0.0f ? -1.0f : (-half_b - sqrt(discriminant)) / a;
}
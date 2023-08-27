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
    //__device__ virtual ~Sphere();

    /**
     * @brief check if a ray hits the sphere or not using the simplified quadratic equation
     */
    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const override;

    /**
     * @brief get the uv coordinates of a point on the sphere
     * @param p the point on the sphere
     * @param u the u coordinate
     * @param v the v coordinate
     */
    __device__ void getSphereUV(const Point3D &p, float &u, float &v) const;
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
    getSphereUV(outward_normal, rec.u, rec.v);
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

__device__ void Sphere::getSphereUV(const Point3D &p, float &u, float &v) const
{
    // <1 0 0> yields <0.5 0.5>
    // <-1 0 0> yields <0 0.5>
    // <0 1 0> yields <0.25 0.5>
    // <0 -1 0> yields <0.75 0.5>
    // <0 0 1> yields <0.5 1>
    // <0 0 -1> yields <0.5 0>

    float theta = acos(-p.y);
    float phi = atan2(-p.z, p.x) + PI;

    u = phi / (2.0f * PI);
    v = theta / PI;
}
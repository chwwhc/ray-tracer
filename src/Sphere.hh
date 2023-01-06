#pragma once

#include "Hittable.hh"
#include "Vec3.hh"

struct Sphere : Hittable
{
    Point3D center;
    double radius;
    shared_ptr<Material> mat_ptr;

    Sphere() = default;
    Sphere(Point3D center, double radius, shared_ptr<Material> mat_ptr) : center(center), radius(radius), mat_ptr(std::move(mat_ptr)) {}

    virtual bool hit(const Ray &r, double t_min, double t_max, HitRecord &rec) const override;
};

bool Sphere::hit(const Ray &r, double t_min, double t_max, HitRecord &rec) const
{
    Vec3 oc = r.origin - center;
    double a = r.direction.squaredLength();
    double half_b = dotProd(oc, r.direction);
    double c = oc.squaredLength() - radius * radius;

    double discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0)
        return false;
    double sqrtd = std::sqrt(discriminant);

    // find the nearest root that lies in the acceptable range
    double root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max)
    {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    rec.normal = (rec.p - center) / radius;
    Vec3 outward_normal = (rec.p - center) / radius;
    rec.setFaceNormal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

// check if a ray hits the sphere or not
// using simplified quadratic equation
double hitSphere(const Point3D &center, double radius, const Ray &r)
{
    Vec3 oc = r.origin - center;
    double a = r.direction.squaredLength();
    double half_b = dotProd(oc, r.direction);
    double c = oc.squaredLength() - radius * radius;
    double discriminant = half_b * half_b - a * c;

    return discriminant < 0 ? -1.0 : (-half_b - std::sqrt(discriminant)) / a;
}
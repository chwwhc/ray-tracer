#pragma once

#include "Util.cuh"
#include "Hittable.cuh"
#include "HittableList.cuh"

class Quad : public Hittable
{
    Point3D Q;
    Vec3 u, v;
    float D;
    Vec3 normal;
    Vec3 w;
    Material *mat;

public:
    __device__ Quad(Point3D Q, Vec3 u, Vec3 v, Material *mat);
    __device__ virtual bool isInterior(float a, float b, HitRecord &rec) const;
    __device__ static HittableList* box(const Point3D& a, const Point3D& b, Material* mat);
    __device__ bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const override;
};

__device__ Quad::Quad(Point3D Q, Vec3 u, Vec3 v, Material *mat)
    : Q(Q), u(u), v(v), mat(mat)
{
    Vec3 n = crossProd(u, v);
    normal = unitVec(n);
    D = dotProd(normal, Q);
    w = n / dotProd(n, n);
}

__device__ bool Quad::isInterior(float a, float b, HitRecord &rec) const
{
    // Given the hit point in plane coordinates, return false if it is outside the
    // primitive, otherwise set the hit record UV coordinates and return true

    if (a < 0.0f || a > 1.0f || b < 0.0f || b > 1.0f)
    {
        return false;
    }

    rec.u = a;
    rec.v = b;
    return true;
}

__device__ bool Quad::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const
{
    float denominator = dotProd(normal, r.direction);

    // No hit if ray is parallel to plane
    if (fabs(denominator) < 1e-8)
    {
        return false;
    }

    // Return false if the hit point is outside the ray range
    float t = (D - dotProd(normal, r.origin)) / denominator;
    if (t < t_min || t > t_max)
    {
        return false;
    }

    // Determine the hit point lies within the planar shape using its plane coordinates
    Point3D intersection = r.at(t);
    Vec3 planar_hitpoint_vec = intersection - Q;
    float alpha = dotProd(w, crossProd(planar_hitpoint_vec, v));
    float beta = dotProd(w, crossProd(u, planar_hitpoint_vec));
    if (!isInterior(alpha, beta, rec))
    {
        return false;
    }

    // Ray hits the 2D shape; set the rest of the hit record and return true
    rec.t = t;
    rec.hit_point = intersection;
    rec.mat_ptr = mat;
    rec.setFaceNormal(r, normal);

    return true;
}

__device__ HittableList* Quad::box(const Point3D& a, const Point3D& b, Material* mat)
{
    
    Hittable** sides = new Hittable * [6];

    Point3D min = Point3D(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
    Point3D max = Point3D(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));

    Vec3 dx = Vec3(max.x - min.x, 0.0f, 0.0f);
    Vec3 dy = Vec3(0.0f, max.y - min.y, 0.0f);
    Vec3 dz = Vec3(0.0f, 0.0f, max.z - min.z);

    sides[0] = new Quad(Point3D(min.x, min.y, min.z), dx, dy, mat); // front
    sides[1] = new Quad(Point3D(max.x, min.y, max.z), -dz, dy, mat); // right
    sides[2] = new Quad(Point3D(max.x, min.y, min.z), -dx, dy, mat); // back
    sides[3] = new Quad(Point3D(min.x, min.y, min.z), dz, dy, mat); // left
    sides[4] = new Quad(Point3D(min.x, max.y, max.z), dx, -dz, mat); // top
    sides[5] = new Quad(Point3D(min.x, min.y, min.z), dx, dz, mat); // bottom

    return new HittableList(sides, 6);
}
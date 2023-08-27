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
    Material *mat_ptr;
    float t;
    float u;
    float v;
    bool is_front_face;

    __device__ void setFaceNormal(const Ray &r, const Vec3 &outward_normal)
    {
        is_front_face = dotProd(r.direction, outward_normal) < 0.0f;
        normal = is_front_face ? outward_normal : -outward_normal;
    }
};

// Hittable interface
struct Hittable
{
    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const = 0;
    //__device__ virtual ~Hittable() {}
};

class Translate : public Hittable
{
    Hittable *object;
    Vec3 offset;

public:
    __device__ Translate(Hittable *object, Vec3 offset) : object(object), offset(offset) {}

    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const override
    {
        Ray moved_r(r.origin - offset, r.direction);
        if (!object->hit(moved_r, t_min, t_max, rec))
        {
            return false;
        }

        rec.hit_point += offset;

        return true;
    }
};

class RotateY : public Hittable
{
    Hittable *object;
    float sin_theta;
    float cos_theta;

public:
    __device__ RotateY(Hittable *object, float angle)
        : object(object)
    {
        float radians = degreesToRadians(angle);
        sin_theta = sinf(radians);
        cos_theta = cosf(radians);
    }

    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const override
    {
        Point3D origin = r.origin;
        Vec3 direction = r.direction;

        origin.x = cos_theta * r.origin.x - sin_theta * r.origin.z;
        origin.z = sin_theta * r.origin.x + cos_theta * r.origin.z;

        direction.x = cos_theta * r.direction.x - sin_theta * r.direction.z;
        direction.z = sin_theta * r.direction.x + cos_theta * r.direction.z;

        Ray rotated_r(origin, direction);

        if (!object->hit(rotated_r, t_min, t_max, rec))
        {
            return false;
        }

        Point3D hit_point = rec.hit_point;
        Vec3 normal = rec.normal;

        hit_point.x = cos_theta * rec.hit_point.x + sin_theta * rec.hit_point.z;
        hit_point.z = -sin_theta * rec.hit_point.x + cos_theta * rec.hit_point.z;

        normal.x = cos_theta * rec.normal.x + sin_theta * rec.normal.z;
        normal.z = -sin_theta * rec.normal.x + cos_theta * rec.normal.z;

        rec.hit_point = hit_point;
        rec.setFaceNormal(rotated_r, normal);

        return true;
    }
};
#pragma once

#include "Hittable.cuh"

class HittableList : public Hittable
{
    Hittable **obj_list;
    size_t list_size;

public:
    __device__ HittableList() = default;
    __device__ HittableList(Hittable **input_obj_list, size_t input_list_size) : obj_list(input_obj_list), list_size(input_list_size) {}
    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const override;
};

__device__ bool HittableList::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const
{
    HitRecord tmp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < list_size; ++i)
    {
        if (obj_list[i]->hit(r, t_min, closest_so_far, tmp_rec))
        {
            hit_anything = true;
            closest_so_far = tmp_rec.t;
            rec = tmp_rec;
        }
    }

    return hit_anything;
}
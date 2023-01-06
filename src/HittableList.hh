#pragma once

#include "Hittable.hh"

#include <memory>
#include <vector>

using std::make_shared;
using std::shared_ptr;

class HittableList : Hittable
{
    std::vector<shared_ptr<Hittable>> objects;

public:
    HittableList() = default;
    HittableList(shared_ptr<Hittable> object)
    {
        objects.emplace_back(object);
    }

    void clear()
    {
        objects.clear();
    }

    void add(shared_ptr<Hittable> object)
    {
        objects.emplace_back(object);
    }

    virtual bool hit(const Ray &r, double t_min, double t_max, HitRecord &rec) const override;
};

bool HittableList::hit(const Ray &r, double t_min, double t_max, HitRecord &rec) const
{
    HitRecord tmp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;

    for (const auto &object : objects)
    {
        if (object->hit(r, t_min, closest_so_far, tmp_rec))
        {
            hit_anything = true;
            closest_so_far = tmp_rec.t;
            rec = tmp_rec;
        }
    }

    return hit_anything;
}
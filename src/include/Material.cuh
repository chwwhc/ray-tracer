#pragma once

#include "Util.cuh"
#include "Hittable.cuh"
#include "Texture.cuh"

struct HitRecord;

struct Material
{
    /**
     * r_in: in-ray
     * attenuation: reflection rate of each color channel
     */
    __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered, curandState* rand_state) const = 0;

    __device__ virtual Color emitted(float u, float v, const Point3D &p) const
    {
        return Color(0.0f, 0.0f, 0.0f);
    }

    //__device__ virtual ~Material() {}
};

struct Lambertian : Material
{
    Texture* albedo; // reflection rate

    __device__ Lambertian(Color albedo) : albedo(new SolidColor(albedo)) {}

    __device__ Lambertian(Texture* albedo) : albedo(albedo) {}

    //__device__ ~Lambertian() override

    __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered, curandState* rand_state) const override
    {
        Vec3 scatter_direction = rec.normal + randomUnitVector(rand_state);

        // catch degenerate scatter direction
        if (scatter_direction.nearZero())
            scatter_direction = rec.normal;

        scattered = Ray(rec.hit_point, scatter_direction);
        attenuation = albedo->value(rec.u, rec.v, rec.hit_point);
        return true;
    }
};

struct Metal : Material
{
    Color albedo;
    float fuzz;

    __device__ Metal(const Color &albedo, float fuzz) : albedo(albedo), fuzz(fuzz < 1.0f ? fuzz : 1.0f) {}

    //__device__ ~Metal() override {}

    __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered, curandState* rand_state) const override
    {
        Vec3 reflected = reflect(unitVec(r_in.direction), rec.normal);
        scattered = Ray(rec.hit_point, reflected + fuzz * randomInUnitSphere(rand_state));
        attenuation = albedo;
        return dotProd(scattered.direction, rec.normal) > 0.0f;
    }
};

class Dielectric : public Material
{
    __device__ static inline float reflectance(float cos, float idx_refra)
    {
        // use Schlick's approximation for reflectance
        float r0 = (1.0f - idx_refra) / (1.0f + idx_refra);
        r0 *= r0;
        return r0 + (1.0f - r0) * pow((1.0f - cos), 5.0f);
    }

public:
    float idx_refra; // index of refraction

    __device__ Dielectric(float idx_refra) : idx_refra(idx_refra) {}

    //__device__ ~Dielectric() override {}

    __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered, curandState* rand_state) const override
    {
        attenuation = Color(1.0f, 1.0f, 1.0f);
        float refraction_ratio = rec.is_front_face ? (1.0f / idx_refra) : idx_refra;

        Vec3 unit_dir = unitVec(r_in.direction);
        float cos_theta = fmin(dotProd(-unit_dir, rec.normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        Vec3 dir = (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(rand_state)) ? reflect(unit_dir, rec.normal) : refract(unit_dir, rec.normal, refraction_ratio);

        scattered = Ray(rec.hit_point, dir);
        return true;
    }
};

class DiffuseLight : public Material
{
    Texture* emit;

public:
    __device__ DiffuseLight(Texture* emit) : emit(emit) {}

    __device__ DiffuseLight(Color c) : emit(new SolidColor(c)) {}

    //__device__ ~DiffuseLight() override {}

    __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered, curandState* rand_state) const override
    {
        return false;
    }

    __device__ virtual Color emitted(float u, float v, const Point3D &p) const override
    {
        return emit->value(u, v, p);
    }
};
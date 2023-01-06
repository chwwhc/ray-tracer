#pragma once

#include "Util.hh"
#include "Hittable.hh"

struct HitRecord;

struct Material
{
    /**
     * r_in: in-ray
     * attenuation: reflection rate of each color channel
     */
    virtual bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered) const = 0;
};

struct Lambertian : Material
{
    Color albedo; // reflection rate

    Lambertian(const Color &albedo) : albedo(albedo) {}

    virtual bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered) const override
    {
        Vec3 scatter_direction = rec.normal + randomUnitVector();

        // catch degenerate scatter direction
        if (scatter_direction.nearZero())
            scatter_direction = rec.normal;

        scattered = Ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

struct Metal : Material
{
    Color albedo;
    double fuzz;

    Metal(const Color &albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1.0 ? fuzz : 1.0) {}

    virtual bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered) const override
    {
        Vec3 reflected = reflect(unitVec(r_in.direction), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz * randomInUnitSphere());
        attenuation = albedo;
        return dotProd(scattered.direction, rec.normal) > 0;
    }
};

class Dielectric : public Material
{
    static inline double reflectance(double cos, double idx_refra)
    {
        // use Schlick's approximation for reflectance
        double r0 = (1 - idx_refra) / (1 + idx_refra);
        r0 *= r0;
        return r0 + (1 - r0) * std::pow((1 - cos), 5);
    }

public:
    double idx_refra; // index of refraction

    Dielectric(double idx_refra) : idx_refra(idx_refra) {}

    virtual bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered) const override
    {
        attenuation = Color(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0 / idx_refra) : idx_refra;

        Vec3 unit_dir = unitVec(r_in.direction);
        double cos_theta = std::fmin(dotProd(-unit_dir, rec.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        Vec3 dir = (cannot_refract || reflectance(cos_theta, refraction_ratio) > randomDouble()) ? reflect(unit_dir, rec.normal) : refract(unit_dir, rec.normal, refraction_ratio);

        scattered = Ray(rec.p, dir);
        return true;
    }
};
#include "HittableList.cuh"
#include "Material.cuh"
#include "Sphere.cuh"
#include "Vec3.cuh"
#include "Camera.cuh"

#define RAND (curand_uniform(&local_rand_state))

__global__ void randomScene(Hittable **obj_list, Hittable **world, Camera **cam, const int image_width, const int image_height, curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // camera parameters
        Point3D lookfrom(13.0f, 2.0f, 3.0f);
        Point3D lookat(0.0f, 0.0f, 0.0f);
        Vec3 vup(0.0f, 1.0f, 0.0f);
        constexpr float aperture = 0.1f;
        constexpr float vfov = 20.0f;
        constexpr float dist_to_focus = 10.0f;
        const float aspect_ratio = static_cast<float>(image_width) / static_cast<float>(image_height);

        curandState local_rand_state(*rand_state);

        obj_list[0] = new Sphere(Point3D(0.0f, -1000.0f, -1.0f), 1000.0f, new Lambertian(Color(0.5f, 0.5f, 0.5f)));

        int i = 1;
        for (int a = -11; a < 11; ++a)
        {
            for (int b = -11; b < 11; ++b)
            {
                float choose_mat = RAND;
                Point3D center(a + RAND, 0.2f, b + RAND);

                if (choose_mat < 0.4f)
                {
                    obj_list[i++] = new Sphere(center, 0.2f, new Lambertian(Color(RAND * RAND, RAND * RAND, RAND * RAND)));
                }
                else if (choose_mat < 0.75f)
                {
                    obj_list[i++] = new Sphere(center, 0.2f, new Metal(Color(0.5f * (1.0f + RAND), 0.5f * (1.0f + RAND), 0.5f * (1.0f + RAND)), 0.5f * RAND));
                }
                else
                {
                    obj_list[i++] = new Sphere(center, 0.2f, new Dielectric(0.7f + RAND));
                }
            }
        }

        obj_list[i++] = new Sphere(Point3D(0.0f, 1.0f, 0.0f), 1.0f, new Dielectric(1.5f));
        obj_list[i++] = new Sphere(Point3D(-4.0f, 1.0f, 0.0f), 1.0f, new Lambertian(Color(0.4f, 0.2f, 0.1f)));
        obj_list[i++] = new Sphere(Point3D(4.0f, 1.0f, 0.0f), 1.0f, new Metal(Color(0.7f, 0.6f, 0.5f), 0.0f));
        *rand_state = local_rand_state;
        *world = new HittableList(obj_list, 22u * 22u + 1u + 3u);
        *cam = new Camera(vfov, aspect_ratio, aperture, dist_to_focus, lookfrom, lookat, vup);
    }
}

__global__ void worldclear(Hittable **obj_list, Hittable **world, Camera **cam)
{
    for (size_t i = 0u; i < 22u * 22u + 1u + 3u; ++i)
    {
        delete ((Sphere *)obj_list[i])->mat_ptr;
        delete obj_list[i];
    }
    delete *world;
    delete *cam;
}
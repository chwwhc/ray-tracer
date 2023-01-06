#include "Util.hh"
#include "Camera.hh"
#include "Color.hh"
#include "HittableList.hh"
#include "Sphere.hh"
#include "WorldGen.hh"
#include "Material.hh"

#include <iostream>
#include <sstream>
#include <fstream>

Color rayColor(const Ray &r, const HittableList &world, int depth)
{
    HitRecord rec;

    // If we've exceeded the ray bounce limit, no more light is gathered
    if (depth <= 0)
        return Color(0, 0, 0);

    if (world.hit(r, 0.001, INF, rec))
    {
        Ray scattered;
        Color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * rayColor(scattered, world, depth - 1);
        return {0, 0, 0};
    }

    Vec3 unit_direction = unitVec(r.direction);
    double t = 0.5 * (unit_direction.y + 1.0);

    return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
}

int main()
{
    // image size
    constexpr double aspect_ratio = 3.0 / 2.0;
    constexpr int image_width = 1200;
    constexpr int image_height = static_cast<int>(image_width / aspect_ratio);
    constexpr int samples_per_pixel = 500;
    constexpr int max_depth = 50;

    // world
    HittableList world = randomScene();

/*
    auto material_ground = make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
    auto material_center = make_shared<Lambertian>(Color(0.7, 0.3, 0.3));
    auto material_left = make_shared<Metal>(Color(0.8, 0.8, 0.8), 0.3);
    auto material_right = make_shared<Metal>(Color(0.8, 0.6, 0.2), 1.0);

    world.add(make_shared<Sphere>(Point3D(0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<Sphere>(Point3D(0.0, 0.0, -1.0), 0.5, material_center));
    world.add(make_shared<Sphere>(Point3D(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add(make_shared<Sphere>(Point3D(1.0, 0.0, -1.0), 0.5, material_right));
*/

    // canvas
    pixelRGB canvas[image_width * image_height];

    // camera
    Point3D lookfrom(13, 2, 3);
    Point3D lookat(0, 0, 0);
    Vec3 vup(0, 1, 0);
    double dist_to_focus = 10.0;
    double aperture = 0.1;
    Camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    // render
    std::ofstream output("image.ppm");
    output << "P3\n"
           << image_width << " " << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j)
    {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i)
        {
            Color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s)
            {
                double u = (i + randomDouble()) / (image_width - 1);
                double v = (j + randomDouble()) / (image_height - 1);
                Ray r = cam.getRay(u, v);
                pixel_color += rayColor(r, world, max_depth);
            }
            writeColor(output, canvas, i, j, image_width, image_height, pixel_color, samples_per_pixel);
        }
    }

    /*
        for (int j = 0; j < image_height; ++j)
        {
            //  std::cerr << "\rScanlines remaining: " << j << " " << std::flush;
            for (int i = 0; i < image_width; ++i)
            {
                int idx = i * image_width + j;
                output << canvas[idx].R << " " << canvas[idx].G << " " << canvas[idx].B << std::endl;
            }
        }
        */

    std::cerr << "\nDone.\n";
}
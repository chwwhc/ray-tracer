#include "HittableList.cuh"
#include "Material.cuh"
#include "Sphere.cuh"
#include "Quad.cuh"
#include "Vec3.cuh"
#include "Camera.cuh"
#include "Texture.cuh"

// scene options
enum SCENE_OPTIONS
{
    RANDOM_SCENE,
    TWO_SPHERE,
    PERLIN_SPHERE,
    EARTH,
    QUADS,
    SIMPLE_LIGHT,
    CORNELL_BOX,
    FINAL_SCENE
};

class SceneConfig
{
    SceneConfig() = default;

    SceneConfig(float aspect_ratio, int image_width, size_t num_hittables, size_t num_mats, const char *output_name = "output.png", Color background = Color(0.0f, 0.0f, 0.0f), int num_sample_per_pixel = 1024, int max_depth = 15)
        : aspect_ratio(aspect_ratio), image_width(image_width), num_hittables(num_hittables), num_mats(num_mats), num_sample_per_pixel(num_sample_per_pixel), max_depth(max_depth), background(background), output_name(output_name)
    {
        image_height = static_cast<int>(image_width / aspect_ratio);
        num_pixel = image_height * image_width;
        frame_buffer_size = num_pixel * 4u * sizeof(uint8_t);
    }

public:
    float aspect_ratio;
    int image_width;
    int image_height;
    int num_pixel;
    int num_sample_per_pixel;
    int max_depth;
    size_t frame_buffer_size;
    size_t num_hittables;
    size_t num_mats;
    Color background;
    const char *output_name;

    static SceneConfig generateSceneConfig(SCENE_OPTIONS scene_option)
    {
        SceneConfig config;

        switch (scene_option)
        {
        case RANDOM_SCENE:
            config = SceneConfig(4.0f / 3.0f, 1200, 22u * 22u + 1u + 3u, 22u * 22u + 1u + 3u, "random_scene.png", Color(0.7f, 0.8f, 1.0f));
            break;
        case TWO_SPHERE:
            config = SceneConfig(16.0f / 9.0f, 800, 2u, 1u, "two_sphere.png", Color(0.7f, 0.8f, 1.0f));
            break;
        case PERLIN_SPHERE:
            config = SceneConfig(4.0f / 3.0f, 800, 2u, 2u, "perlin_sphere.png", Color(0.7f, 0.8f, 1.0f));
            break;
        case EARTH:
            config = SceneConfig(1.0f, 400, 1u, 1u, "earth.png", Color(0.7f, 0.8f, 1.0f));
            break;
        case QUADS:
            config = SceneConfig(1.0f, 400, 5u, 5u, "quads.png", Color(0.7f, 0.8f, 1.0f));
            break;
        case SIMPLE_LIGHT:
            config = SceneConfig(16.0f / 9.0f, 1200, 4u, 2u, "simple_light.png");
            break;
        case CORNELL_BOX:
            config = SceneConfig(1.0f, 600, 8u, 4u, "cornell_box.png");
            break;
        case FINAL_SCENE:
            config = SceneConfig(1.0f, 1200, 20u * 20u + 7u, 20u * 20u + 7u, "final_scene.png", Color(0.0f, 0.0f, 0.01f));
            break;
        default:
            throw std::runtime_error("Invalid scene option");
        }

        return config;
    }
};

__global__ void finalScene(Material **mat_list, Hittable **obj_list, Hittable **world, Camera **cam, curandState *rand_state, uint8_t **images, int *tex_image_widths, int *tex_image_heights, SceneConfig *config)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state(*rand_state);

        // ground boxes
        int boxes_per_side = 20;
        for (int i = 0; i < boxes_per_side; ++i)
        {
            for (int j = 0; j < boxes_per_side; ++j)
            {
                float w = 100.0f;
                float x0 = -1000.0f + i * w;
                float z0 = -1000.0f + j * w;
                float y0 = 0.0f;
                float x1 = x0 + w;
                float y1 = 100.0f * (RAND + 0.01f);
                float z1 = z0 + w;
                Material *rand_mat = new Lambertian(Color(RAND * RAND, RAND * RAND, RAND * RAND));
                mat_list[i * boxes_per_side + j] = rand_mat;
                obj_list[i * boxes_per_side + j] = Quad::box(Point3D(x0, y0, z0), Point3D(x1, y1, z1), rand_mat);
            }
        }

        // light
        Material *light = new DiffuseLight(Color(7.0f, 7.0f, 7.0f));
        mat_list[boxes_per_side * boxes_per_side] = light;
        obj_list[boxes_per_side * boxes_per_side] = new Quad(Point3D(123.0f, 554.0f, 147.0f), Vec3(0.0f, 0.0f, 654.0f), Vec3(654.0f, 0.0f, 0.0f), light);

        // sphere
        Point3D center1(400.0f, 400.0f, 200.0f);
        Point3D center2 = center1 + Vec3(-150.0f, 0.0f, 0.0f);
        Point3D center3 = center1 + Vec3(-300.0f, -30.0f, 10.0f);
        Material *glass = new Dielectric(1.5f);
        Material *pink = new Lambertian(Color(0.8f, 0.3f, 0.3f));
        Material *metal = new Metal(Color(0.8f, 0.6f, 0.2f), 0.0f);
        mat_list[boxes_per_side * boxes_per_side + 1] = glass;
        mat_list[boxes_per_side * boxes_per_side + 2] = pink;
        mat_list[boxes_per_side * boxes_per_side + 3] = metal;
        obj_list[boxes_per_side * boxes_per_side + 1] = new Sphere(center1, 50.0f, glass);
        obj_list[boxes_per_side * boxes_per_side + 2] = new Sphere(center2, 50.0f, pink);
        obj_list[boxes_per_side * boxes_per_side + 3] = new Sphere(center3, 50.0f, metal);

        // earth
        uint8_t *earth_image = images[0];
        int earth_tex_image_width = tex_image_widths[0];
        int earth_tex_image_height = tex_image_heights[0];

        Material *earth_texture = new Lambertian(new ImageTexture(earth_image, earth_tex_image_width, earth_tex_image_height));
        mat_list[boxes_per_side * boxes_per_side + 4] = earth_texture;
        obj_list[boxes_per_side * boxes_per_side + 4] = new Sphere(Point3D(400.0f, 200.0f, 400.0f), 100.0f, earth_texture);

        // jupiter
        uint8_t *jupiter_image = images[1];
        int jupiter_tex_image_width = tex_image_widths[1];
        int jupiter_tex_image_height = tex_image_heights[1];

        Material *jupiter_texture = new Lambertian(new ImageTexture(jupiter_image, jupiter_tex_image_width, jupiter_tex_image_height));
        mat_list[boxes_per_side * boxes_per_side + 5] = jupiter_texture;
        obj_list[boxes_per_side * boxes_per_side + 5] = new Sphere(Point3D(100.0f, 200.0f, 300.0f), 80.0f, jupiter_texture);

        // perlin sphere
        Material *per_text = new Lambertian(new NoiseTexture(5.0f, &local_rand_state));
        mat_list[boxes_per_side * boxes_per_side + 6] = per_text;
        obj_list[boxes_per_side * boxes_per_side + 6] = new Sphere(Point3D(220.0f, 280.0f, 300.0f), 80.0f, per_text);

        *world = new HittableList(obj_list, config->num_hittables);

        // camera parameters
        Point3D lookfrom(478.0f, 278.0f, -600.0f);
        Point3D lookat(278.0f, 278.0f, 0.0f);
        Vec3 vup(0.0f, 1.0f, 0.0f);
        constexpr float aperture = 0.0f;
        constexpr float vfov = 40.0f;
        constexpr float dist_to_focus = 10.0f;

        *cam = new Camera(vfov, config->aspect_ratio, aperture, dist_to_focus, lookfrom, lookat, vup);
        *rand_state = local_rand_state;
    }
}

__global__ void cornellBox(Material **mat_list, Hittable **obj_list, Hittable **world, Camera **cam, curandState *rand_state, SceneConfig *config)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state(*rand_state);

        // Materials
        Material *red = new Lambertian(Color(0.65f, 0.05f, 0.05f));
        Material *white = new Lambertian(Color(0.73f, 0.73f, 0.73f));
        Material *green = new Lambertian(Color(0.12f, 0.45f, 0.15f));
        Material *light = new DiffuseLight(Color(15.0f, 15.0f, 15.0f));
        mat_list[0] = red;
        mat_list[1] = white;
        mat_list[2] = green;
        mat_list[3] = light;

        // Walls
        obj_list[0] = new Quad(Point3D(555.0f, 0.0f, 0.0f), Vec3(0.0f, 555.0f, 0.0f), Vec3(0.0f, 0.0f, 555.0f), green);
        obj_list[1] = new Quad(Point3D(0.0f, 0.0f, 0.0f), Vec3(0.0f, 555.0f, 0.0f), Vec3(0.0f, 0.0f, 555.0f), red);
        obj_list[2] = new Quad(Point3D(343.0f, 554.0f, 332.0f), Vec3(-130.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, -105.0f), light);
        obj_list[3] = new Quad(Point3D(0.0f, 0.0f, 0.0f), Vec3(555.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 555.0f), white);
        obj_list[4] = new Quad(Point3D(555.0f, 555.0f, 555.0f), Vec3(-555.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, -555.0f), white);
        obj_list[5] = new Quad(Point3D(0.0f, 0.0f, 555.0f), Vec3(555.0f, 0.0f, 0.0f), Vec3(0.0f, 555.0f, 0.0f), white);

        // Boxes
        Hittable *box1 = Quad::box(Point3D(0.0f, 0.0f, 0.0f), Point3D(165.0f, 330.0f, 165.0f), white);
        box1 = new RotateY(box1, 15.0f);
        box1 = new Translate(box1, Vec3(265.0f, 0.0f, 295.0f));
        obj_list[6] = box1;

        Hittable *box2 = Quad::box(Point3D(0.0f, 0.0f, 0.0f), Point3D(165.0f, 165.0f, 165.0f), white);
        box2 = new RotateY(box2, -18.0f);
        box2 = new Translate(box2, Vec3(130.0f, 0.0f, 65.0f));
        obj_list[7] = box2;

        // Hittable list with the five spheres
        *world = new HittableList(obj_list, config->num_hittables);

        // camera parameters
        Point3D lookfrom(278.0f, 278.0f, -800.0f);
        Point3D lookat(278.0f, 278.0f, 0.0f);
        Vec3 vup(0.0f, 1.0f, 0.0f);
        constexpr float aperture = 0.1f;
        constexpr float vfov = 40.0f;
        constexpr float dist_to_focus = 10.0f;

        *cam = new Camera(vfov, config->aspect_ratio, aperture, dist_to_focus, lookfrom, lookat, vup);
        *rand_state = local_rand_state;
    }
}

__global__ void simpleLight(Material **mat_list, Hittable **obj_list, Hittable **world, Camera **cam, curandState *rand_state, SceneConfig *config)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state(*rand_state);

        // Materials
        Material *per_text = new Lambertian(new NoiseTexture(4.0f, rand_state));
        Material *diff_light = new DiffuseLight(Color(4.0f, 4.0f, 4.0f));
        mat_list[0] = per_text;
        mat_list[1] = diff_light;

        // Hittables
        obj_list[0] = new Sphere(Point3D(0.0f, -1000.0f, 0.0f), 1000.0f, per_text);
        obj_list[1] = new Sphere(Point3D(0.0f, 2.0f, 0.0f), 2.0f, per_text);
        obj_list[2] = new Quad(Point3D(3.0f, 1.0f, -2.0f), Vec3(2.0f, 0.0f, 0.0f), Vec3(0.0f, 2.0f, 0.0f), diff_light);
        obj_list[3] = new Sphere(Point3D(0.0f, 7.0f, 0.0f), 2.0f, diff_light);

        // Hittable list with the five spheres
        *world = new HittableList(obj_list, config->num_hittables);

        // camera parameters
        Point3D lookfrom(26.0f, 3.0f, 6.0f);
        Point3D lookat(0.0f, 2.0f, 0.0f);
        Vec3 vup(0.0f, 1.0f, 0.0f);
        constexpr float aperture = 0.1f;
        constexpr float vfov = 20.0f;
        constexpr float dist_to_focus = 10.0f;

        *cam = new Camera(vfov, config->aspect_ratio, aperture, dist_to_focus, lookfrom, lookat, vup);
        *rand_state = local_rand_state;
    }
}

__global__ void quads(Material **mat_list, Hittable **obj_list, Hittable **world, Camera **cam, curandState *rand_state, SceneConfig *config)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state(*rand_state);

        // Materials
        Material *left_red = new Lambertian(Color(1.0f, 0.2f, 0.2f));
        Material *back_green = new Lambertian(Color(0.2f, 1.0f, 0.2f));
        Material *right_blue = new Lambertian(Color(0.2f, 0.2f, 1.0f));
        Material *upper_orange = new Lambertian(Color(1.0f, 0.5f, 0.0f));
        Material *lower_teal = new Lambertian(Color(0.2f, 0.8f, 0.8f));
        mat_list[0] = left_red;
        mat_list[1] = back_green;
        mat_list[2] = right_blue;
        mat_list[3] = upper_orange;
        mat_list[4] = lower_teal;

        // Quads
        obj_list[0] = new Quad(Point3D(-3.0f, -2.0f, 5.0f), Vec3(0.0f, 0.0f, -4.0f), Vec3(0.0f, 4.0f, 0.0f), left_red);
        obj_list[1] = new Quad(Point3D(-2.0f, -2.0f, 0.0f), Vec3(4.0f, 0.0f, 0.0f), Vec3(0.0f, 4.0f, 0.0f), back_green);
        obj_list[2] = new Quad(Point3D(3.0f, -2.0f, 1.0f), Vec3(0.0f, 0.0f, 4.0f), Vec3(0.0f, 4.0f, 0.0f), right_blue);
        obj_list[3] = new Quad(Point3D(-2.0f, 3.0f, 1.0f), Vec3(4.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 4.0f), upper_orange);
        obj_list[4] = new Quad(Point3D(-2.0f, -3.0f, 5.0f), Vec3(4.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, -4.0f), lower_teal);

        // Hittable list with the five quads
        *world = new HittableList(obj_list, config->num_hittables);

        // camera parameters
        Point3D lookfrom(0.0f, 0.0f, 9.0f);
        Point3D lookat(0.0f, 0.0f, 0.0f);
        Vec3 vup(0.0f, 1.0f, 0.0f);
        constexpr float aperture = 0.1f;
        constexpr float vfov = 80.0f;
        constexpr float dist_to_focus = 10.0f;

        *cam = new Camera(vfov, config->aspect_ratio, aperture, dist_to_focus, lookfrom, lookat, vup);
        *rand_state = local_rand_state;
    }
}

__global__ void perlinSphere(Material **mat_list, Hittable **obj_list, Hittable **world, Camera **cam, curandState *rand_state, SceneConfig *config)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state(*rand_state);

        // Materials
        Material *per_text = new Lambertian(new NoiseTexture(4.0f, rand_state));

        // grass land material
        Material *ground = new Lambertian(Color(0.2f, 0.8f, 0.2f));

        mat_list[0] = per_text;
        mat_list[1] = ground;

        // Hittables
        obj_list[0] = new Sphere(Point3D(0.0f, -1000.0f, 0.0f), 1000.0f, ground);
        obj_list[1] = new Sphere(Point3D(0.0f, 3.5f, 0.0f), 3.5f, per_text);

        *world = new HittableList(obj_list, config->num_hittables);

        // camera parameters
        Point3D lookfrom(13.0f, 4.0f, 8.0f);
        Point3D lookat(0.0f, 2.0f, 0.0f);
        Vec3 vup(0.0f, 1.0f, 0.0f);
        constexpr float aperture = 0.1f;
        constexpr float vfov = 40.0f;
        constexpr float dist_to_focus = 10.0f;

        *cam = new Camera(vfov, config->aspect_ratio, aperture, dist_to_focus, lookfrom, lookat, vup);
        *rand_state = local_rand_state;
    }
}

__global__ void earth(Material **mat_list, Hittable **obj_list, Hittable **world, Camera **cam, curandState *rand_state, uint8_t **images, int *tex_image_widths, int *tex_image_heights, SceneConfig *config)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state(*rand_state);

        // Materials
        uint8_t *image = images[0];
        int tex_image_width = tex_image_widths[0];
        int tex_image_height = tex_image_heights[0];

        Material *earth_texture = new Lambertian(new ImageTexture(image, tex_image_width, tex_image_height));
        mat_list[0] = earth_texture;

        // Hittables
        obj_list[0] = new Sphere(Point3D(0.0f, 0.0f, 0.0f), 2.0f, earth_texture);

        *world = new HittableList(obj_list, config->num_hittables);

        // camera parameters
        Point3D lookfrom(13.0f, 2.0f, 3.0f);
        Point3D lookat(0.0f, 0.0f, 0.0f);
        Vec3 vup(0.0f, 1.0f, 0.0f);
        constexpr float aperture = 0.1f;
        constexpr float vfov = 20.0f;
        constexpr float dist_to_focus = 10.0f;

        *cam = new Camera(vfov, config->aspect_ratio, aperture, dist_to_focus, lookfrom, lookat, vup);
        *rand_state = local_rand_state;
    }
}

__global__ void twoSpheres(Material **mat_list, Hittable **obj_list, Hittable **world, Camera **cam, curandState *rand_state, SceneConfig *config)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state(*rand_state);

        // Materials
        Material *checker = new Lambertian(new CheckerTexture(10.0f, Color(0.2f, 0.3f, 0.1f), Color(0.9f, 0.9f, 0.9f)));

        // Hittables
        obj_list[0] = new Sphere(Point3D(0.0f, -10.0f, 0.0f), 10.0f, checker);
        obj_list[1] = new Sphere(Point3D(0.0f, 10.0f, 0.0f), 10.0f, checker);

        *world = new HittableList(obj_list, config->num_hittables);

        // camera parameters
        Point3D lookfrom(13.0f, 2.0f, 3.0f);
        Point3D lookat(0.0f, 0.0f, 0.0f);
        Vec3 vup(0.0f, 1.0f, 0.0f);
        constexpr float aperture = 0.1f;
        constexpr float vfov = 20.0f;
        constexpr float dist_to_focus = 10.0f;

        *cam = new Camera(vfov, config->aspect_ratio, aperture, dist_to_focus, lookfrom, lookat, vup);
        *rand_state = local_rand_state;
    }
}

__global__ void randomScene(Material **mat_list, Hittable **obj_list, Hittable **world, Camera **cam, curandState *rand_state, SceneConfig *config)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state(*rand_state);

        Material *ground = new Lambertian(Color(0.5f, 0.5f, 0.5f));
        mat_list[0] = ground;
        obj_list[0] = new Sphere(Point3D(0.0f, -1000.0f, -1.0f), 1000.0f, ground);

        int i = 1;
        for (int a = -11; a < 11; ++a)
        {
            for (int b = -11; b < 11; ++b)
            {
                float choose_mat = RAND;
                Point3D center(a + RAND, 0.2f, b + RAND);
                Material *albedo;

                if (choose_mat < 0.6f)
                {
                    albedo = new Lambertian(Color(RAND * RAND, RAND * RAND, RAND * RAND));
                }
                else if (choose_mat < 0.85f)
                {
                    albedo = new Metal(Color(0.5f * (1.0f + RAND), 0.5f * (1.0f + RAND), 0.5f * (1.0f + RAND)), 0.5f * RAND);
                }
                else
                {
                    albedo = new Dielectric(0.7f + RAND);
                }

                obj_list[i] = new Sphere(center, 0.2f, albedo);
                mat_list[i] = albedo;
                i += 1;
            }
        }

        Material *mat1 = new Dielectric(1.5f);
        mat_list[i] = mat1;
        obj_list[i++] = new Sphere(Point3D(0.0f, 1.0f, 0.0f), 1.0f, mat1);

        Material *mat2 = new Lambertian(Color(0.4f, 0.2f, 0.1f));
        mat_list[i] = mat2;
        obj_list[i++] = new Sphere(Point3D(-4.0f, 1.0f, 0.0f), 1.0f, mat2);

        Material *mat3 = new Metal(Color(0.7f, 0.6f, 0.5f), 0.0f);
        mat_list[i] = mat3;
        obj_list[i++] = new Sphere(Point3D(4.0f, 1.0f, 0.0f), 1.0f, mat3);

        *world = new HittableList(obj_list, config->num_hittables);

        // camera parameters
        Point3D lookfrom(13.0f, 2.0f, 3.0f);
        Point3D lookat(0.0f, 0.0f, 0.0f);
        Vec3 vup(0.0f, 1.0f, 0.0f);
        constexpr float aperture = 0.1f;
        constexpr float vfov = 20.0f;
        constexpr float dist_to_focus = 10.0f;

        *cam = new Camera(vfov, config->aspect_ratio, aperture, dist_to_focus, lookfrom, lookat, vup);
        *rand_state = local_rand_state;
    }
}

__global__ void worldclear(Hittable **obj_list, Hittable **world, Camera **cam, size_t size)
{
    return;
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (size_t i = 0u; i < size; ++i)
        {
            delete obj_list[i];
        }
        delete *world;
        delete *cam;
    }
}
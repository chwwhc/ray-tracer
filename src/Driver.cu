#define STB_IMAGE_IMPLEMENTATION

#include "extern/lodepng.h"
#include "extern/stb_image.h"
#include "include/Util.cuh"
#include "include/Camera.cuh"
#include "include/Sphere.cuh"
#include "include/HittableList.cuh"
#include "include/Material.cuh"
#include "include/WorldGen.cuh"
#include "include/Quad.cuh"

#include <iostream>
#include <thread>
#include <curand_kernel.h>

#define ANTI_ALIASING_KERNEL_BLOCK_SIZE 32

// display progress
bool finish_signal;

// kernel configuration
constexpr int num_thread_x = 8;
constexpr int num_thread_y = 8;

void loadImageTextures(const char **image_paths, int num_textures, uint8_t **&gpu_image_data, int *&gpu_tex_widths, int *&gpu_tex_heights, int bytes_per_pixel = 3)
{
    // Allocate host-side pointer array
    uint8_t **host_gpu_image_data = new uint8_t *[num_textures];
    int *host_widths = new int[num_textures];
    int *host_heights = new int[num_textures];

    // Allocate GPU-side pointer array
    checkCudaErrors(cudaMalloc((void **)&gpu_tex_widths, num_textures * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&gpu_tex_heights, num_textures * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&gpu_image_data, num_textures * sizeof(uint8_t *)));

    for (int i = 0; i < num_textures; ++i)
    {
        uint8_t *host_image_data = stbi_load(image_paths[i], &host_widths[i], &host_heights[i], &bytes_per_pixel, bytes_per_pixel);
        if (host_image_data == nullptr)
        {
            std::cerr << "Error loading image at " << image_paths[i] << std::endl;
            std::exit(EXIT_FAILURE);
        }
        size_t image_size = host_widths[i] * host_heights[i] * bytes_per_pixel * sizeof(uint8_t);
        checkCudaErrors(cudaMalloc((void **)&host_gpu_image_data[i], image_size));
        checkCudaErrors(cudaMemcpy(host_gpu_image_data[i], host_image_data, image_size, cudaMemcpyHostToDevice));
        stbi_image_free(host_image_data);
    }

    // Copy GPU pointers to the GPU-side pointer array
    checkCudaErrors(cudaMemcpy(gpu_tex_widths, host_widths, num_textures * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gpu_tex_heights, host_heights, num_textures * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gpu_image_data, host_gpu_image_data, num_textures * sizeof(uint8_t *), cudaMemcpyHostToDevice));

    // Free the host-side pointer array
    delete[] host_gpu_image_data;
}

/**
 * @brief Convert a float to a uint8_t in sRGB color space
 * @param x The float to be converted
 * @return The converted uint8_t
 */
__device__ inline uint8_t toSRGB(float x)
{
    if (x <= 0.0031308f)
    {
        float result = clamp(x * 12.92f * 255.99f, 0.0f, 255.0f);
        return static_cast<uint8_t>(result);
    }
    else
    {
        float result = clamp((1.055f * pow(x, 1.0f / 2.4f) - 0.055f) * 255.99f, 0.0f, 255.0f);
        return static_cast<uint8_t>(result);
    }
}

__device__ Color rayColor(const Ray &r, Hittable **world, int depth, curandState *rand_state, Color background)
{
    Ray curr_ray = r;
    Color curr_attenuation(1.0f, 1.0f, 1.0f);
    Color final_color(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < depth; ++i)
    {
        HitRecord rec;

        // If the ray hits nothing, break and add the background color
        if (!((*world)->hit(curr_ray, 0.001f, FLT_MAX, rec)))
        {
            final_color += curr_attenuation * background; // Apply current attenuation to the background
            break;
        }

        Ray scattered;
        Color attenuation;
        Color color_from_emission = rec.mat_ptr->emitted(rec.u, rec.v, rec.hit_point);

        if (!rec.mat_ptr->scatter(curr_ray, rec, attenuation, scattered, rand_state))
        {
            final_color += curr_attenuation * color_from_emission; // Apply current attenuation to the emission
            break;
        }

        curr_attenuation *= attenuation; // Multiply the current attenuation by this step's attenuation
        curr_ray = scattered;
    }

    return final_color;
}

__global__ void randInit(curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curand_init(1989ull, 0ull, 0ull, rand_state);
    }
}

__global__ void renderInit(curandState *rand_state, const int image_width, const int image_height)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= image_width || j >= image_height)
        return;

    const int pixel_idx = j * image_width + i;
    curand_init(1989ull + static_cast<uint64_t>(pixel_idx), 0ull, 0ull, &rand_state[pixel_idx]);
}

__global__ void antiAliasingKernel(int i, int j, int max_depth, int num_sample, int image_width, int image_height, Camera **cam, Hittable **world, curandState *local_rand_state, Color *color, Color background)
{
    __shared__ float sharedColor[3][ANTI_ALIASING_KERNEL_BLOCK_SIZE];
    int s = threadIdx.x + blockIdx.x * blockDim.x;

    if (s >= num_sample)
    {
        return;
    }

    Color curr_color(0.0f, 0.0f, 0.0f);

    const float u = (static_cast<float>(i) + curand_uniform(local_rand_state)) / static_cast<float>(image_width);
    const float v = (static_cast<float>(j) + curand_uniform(local_rand_state)) / static_cast<float>(image_height);
    Ray r = (*cam)->getRay(u, v, local_rand_state);
    curr_color = rayColor(r, world, max_depth, local_rand_state, background);

    sharedColor[0][threadIdx.x] = curr_color[0];
    sharedColor[1][threadIdx.x] = curr_color[1];
    sharedColor[2][threadIdx.x] = curr_color[2];
    __syncthreads();

    // Perform a reduction within the warp
    for (int stride = warpSize / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            sharedColor[0][threadIdx.x] += sharedColor[0][threadIdx.x + stride];
            sharedColor[1][threadIdx.x] += sharedColor[1][threadIdx.x + stride];
            sharedColor[2][threadIdx.x] += sharedColor[2][threadIdx.x + stride];
        }
        __syncwarp();
    }

    // Only one atomic add operation per block
    if (threadIdx.x == 0)
    {
        atomicAdd(&(*color)[0], sharedColor[0][0]);
        atomicAdd(&(*color)[1], sharedColor[1][0]);
        atomicAdd(&(*color)[2], sharedColor[2][0]);
    }
}

__global__ void render(uint8_t *frame_buffer, int max_depth, int image_width, int image_height, int num_sample, Camera **cam, Hittable **world, curandState *rand_state, Color background)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= image_width || j >= image_height)
    {
        return;
    }

    int pixel_idx = (image_height - 1) * image_width - j * image_width + i;
    curandState *local_rand_state = new curandState(rand_state[pixel_idx]);
    Color *color = new Color(0.0f, 0.0f, 0.0f);

    // anti-aliasing: take the average of all the samples
    antiAliasingKernel<<<(num_sample + ANTI_ALIASING_KERNEL_BLOCK_SIZE - 1) / ANTI_ALIASING_KERNEL_BLOCK_SIZE, ANTI_ALIASING_KERNEL_BLOCK_SIZE>>>(i, j, max_depth, num_sample, image_width, image_height, cam, world, local_rand_state, color, background);
    cudaDeviceSynchronize();

    rand_state[pixel_idx] = *local_rand_state;
    *color /= static_cast<float>(num_sample);
    int ir = toSRGB((*color)[0]);
    int ig = toSRGB((*color)[1]);
    int ib = toSRGB((*color)[2]);

    frame_buffer[pixel_idx * 4 + 0] = ir;
    frame_buffer[pixel_idx * 4 + 1] = ig;
    frame_buffer[pixel_idx * 4 + 2] = ib;
    frame_buffer[pixel_idx * 4 + 3] = 255;

    delete color;
    delete local_rand_state;
}

void displayProgress(const char *message, bool &finish_signal)
{
    finish_signal = false;
    while (!finish_signal)
    {
        std::cout << "\rGPU is " << message << " \\" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::cout << "\rGPU is " << message << " |" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::cout << "\rGPU is " << message << " /" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::cout << "\rGPU is " << message << " -" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::cout << "\rGPU finished " << message << std::endl;
}

int main()
{
    // Display information
    std::thread wait_message;
    clock_t start, stop;
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    std::cerr << "========================================================================\nNVIDIA GPU Information\n========================================================================\n";
    for (int i = 0; i < num_devices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cerr << "Device Number: " << i << "\n"
                  << "    Device Name: " << prop.name << "\n"
                  << "    Memory Clock Rate (KHz): " << prop.memoryClockRate << "\n"
                  << "    Memory Bus Width (bits): " << prop.memoryBusWidth << "\n"
                  << "    Peak Memory Bandwidth (GB/s): " << 2.0f * static_cast<float>(prop.memoryClockRate) * (static_cast<float>(prop.memoryBusWidth) / 8.0f) / 1.0e6f << "\n\n";
    }

    // GPU random state preparation
    curandState *rand_state_world;
    checkCudaErrors(cudaMalloc((void **)&rand_state_world, sizeof(curandState)));

    wait_message = std::thread(displayProgress, "preparing for random scene generation", std::ref(finish_signal));
    randInit<<<1, 1>>>(rand_state_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    finish_signal = true;
    wait_message.join();

    // scene configuration
    SCENE_OPTIONS scene_option = SCENE_OPTIONS::FINAL_SCENE;
    SceneConfig scene_config = SceneConfig::generateSceneConfig(scene_option);
    SceneConfig *scene_config_gpu;

    // generate the world on the GPU
    Hittable **obj_list;
    Material **mat_list;
    Hittable **world;
    Camera **cam;

    // image texture buffer
    int num_textures;
    uint8_t **gpu_image_data;
    int *gpu_tex_widths;
    int *gpu_tex_heights;

    checkCudaErrors(cudaMalloc((void **)&obj_list, scene_config.num_hittables * sizeof(Hittable *)));
    checkCudaErrors(cudaMalloc((void **)&mat_list, scene_config.num_mats * sizeof(Material *)));
    checkCudaErrors(cudaMalloc((void **)&world, sizeof(Hittable *)));
    checkCudaErrors(cudaMalloc((void **)&cam, sizeof(Camera *)));

    // scene configuration
    checkCudaErrors(cudaMalloc((void **)&scene_config_gpu, sizeof(SceneConfig)));
    checkCudaErrors(cudaMemcpy(scene_config_gpu, &scene_config, sizeof(SceneConfig), cudaMemcpyHostToDevice));

    wait_message = std::thread(displayProgress, "generating the world", std::ref(finish_signal));

    switch (scene_option)
    {
    case SCENE_OPTIONS::TWO_SPHERE:
        twoSpheres<<<1, 1>>>(mat_list, obj_list, world, cam, rand_state_world, scene_config_gpu);
        break;
    case SCENE_OPTIONS::RANDOM_SCENE:
        randomScene<<<1, 1>>>(mat_list, obj_list, world, cam, rand_state_world, scene_config_gpu);
        break;
    case SCENE_OPTIONS::EARTH:
    {
        // image texture buffer
        const char *image_paths[] = {"./images/earthmap.jpg"};
        num_textures = 1;
        loadImageTextures(image_paths, num_textures, gpu_image_data, gpu_tex_widths, gpu_tex_heights);

        earth<<<1, 1>>>(mat_list, obj_list, world, cam, rand_state_world, gpu_image_data, gpu_tex_widths, gpu_tex_heights, scene_config_gpu);
        break;
    }
    case SCENE_OPTIONS::PERLIN_SPHERE:
        perlinSphere<<<1, 1>>>(mat_list, obj_list, world, cam, rand_state_world, scene_config_gpu);
        break;
    case SCENE_OPTIONS::QUADS:
        quads<<<1, 1>>>(mat_list, obj_list, world, cam, rand_state_world, scene_config_gpu);
        break;
    case SCENE_OPTIONS::SIMPLE_LIGHT:
        simpleLight<<<1, 1>>>(mat_list, obj_list, world, cam, rand_state_world, scene_config_gpu);
        break;
    case SCENE_OPTIONS::CORNELL_BOX:
        cornellBox<<<1, 1>>>(mat_list, obj_list, world, cam, rand_state_world, scene_config_gpu);
        break;
    case SCENE_OPTIONS::FINAL_SCENE:
    {
        // image texture buffer
        const char *image_paths[] = {"./images/earthmap.jpg", "./images/jupitermap.jpg"};
        num_textures = 2;
        loadImageTextures(image_paths, num_textures, gpu_image_data, gpu_tex_widths, gpu_tex_heights);

        finalScene<<<1, 1>>>(mat_list, obj_list, world, cam, rand_state_world, gpu_image_data, gpu_tex_widths, gpu_tex_heights, scene_config_gpu);
        break;
    }
    }

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    finish_signal = true;
    wait_message.join();

    // GPU frame buffer
    uint8_t *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, scene_config.frame_buffer_size));

    // launch the rendering kernel
    dim3 dim_grid(scene_config.image_width / num_thread_x + 1, scene_config.image_height / num_thread_y + 1, 1);
    dim3 dim_block(num_thread_x, num_thread_y, 1);
    curandState *rand_state;
    wait_message = std::thread(displayProgress, "preparing for rendering", std::ref(finish_signal));
    checkCudaErrors(cudaMalloc((void **)&rand_state, scene_config.num_pixel * sizeof(curandState)));
    renderInit<<<dim_grid, dim_block>>>(rand_state, scene_config.image_width, scene_config.image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    finish_signal = true;
    wait_message.join();

    // run the render kernel
    wait_message = std::thread(displayProgress, "rendering", std::ref(finish_signal));
    start = clock();
    render<<<dim_grid, dim_block>>>(frame_buffer, scene_config.max_depth, scene_config.image_width, scene_config.image_height, scene_config.num_sample_per_pixel, cam, world, rand_state, scene_config.background);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    finish_signal = true;
    wait_message.join();
    const float timer = (static_cast<float>(stop - start)) / static_cast<float>(CLOCKS_PER_SEC);
    std::cerr << "========================================================================\ntook GPU " << timer << " seconds to render.\n========================================================================\n";

    // save as a png file
    lodepng_encode32_file(scene_config.output_name, frame_buffer, scene_config.image_width, scene_config.image_height);

    // free up GPU memory
    checkCudaErrors(cudaDeviceSynchronize());
    wait_message = std::thread(displayProgress, "cleaning up resources", std::ref(finish_signal));
    worldclear<<<1, 1>>>(obj_list, world, cam, scene_config.num_hittables);
    checkCudaErrors(cudaGetLastError());
    finish_signal = true;
    wait_message.join();

    /*
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(obj_list));
    checkCudaErrors(cudaFree(rand_state));
    checkCudaErrors(cudaFree(frame_buffer));
    checkCudaErrors(cudaFree(scene_config_gpu));
    if (scene_option == SCENE_OPTIONS::EARTH || scene_option == SCENE_OPTIONS::FINAL_SCENE)
    {
        for (int i = 0; i < num_textures; ++i)
        {
            checkCudaErrors(cudaFree(gpu_image_data[i]));
        }
        checkCudaErrors(cudaFree(gpu_tex_widths));
        checkCudaErrors(cudaFree(gpu_tex_heights));
        checkCudaErrors(cudaFree(gpu_image_data));
    }
    */
    cudaDeviceReset();
}
#include "./extern/lodepng.h"
#include "./include/Util.cuh"
#include "./include/Camera.cuh"
#include "./include/Sphere.cuh"
#include "./include/HittableList.cuh"
#include "./include/Material.cuh"
#include "./include/WorldGen.cuh"

#include <iostream>
#include <cfloat>
#include <string>
#include <atomic>
#include <thread>
#include <curand_kernel.h>

// display progress
std::atomic_bool finish_signal;

// image parameters
constexpr float aspect_ratio = 3.0f / 2.0f;
constexpr int image_width = 1200;
constexpr int image_height = static_cast<int>(image_width / aspect_ratio);
constexpr int num_sample_per_pixel = 500;
constexpr int max_depth = 50;
constexpr int num_pixel = image_height * image_width;
constexpr size_t frame_buffer_size = num_pixel * 4u * sizeof(uint8_t);

// kernel configuration
constexpr int num_thread_x = 8;
constexpr int num_thread_y = 8;

__device__ Color rayColor(const Ray &r, Hittable **world, int depth, curandState *rand_state)
{
    Ray curr_ray = r;
    Vec3 curr_attenuation(1.0f, 1.0f, 1.0f);

    for (int i = 0; i < depth; ++i)
    {
        HitRecord rec;
        if ((*world)->hit(curr_ray, 0.001f, FLT_MAX, rec))
        {
            Ray scattered_ray;
            Vec3 attenuation;
            if (rec.mat_ptr->scatter(curr_ray, rec, attenuation, scattered_ray, rand_state))
            {
                curr_attenuation *= attenuation;
                curr_ray = scattered_ray;
            }
            else
                break;
        }
        else
        {
            Vec3 unit_dir = unitVec(curr_ray.direction);
            float t = 0.5f * (unit_dir.y + 1.0f);
            Color c = (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
            return curr_attenuation * c;
        }
    }

    return Color(0.0f, 0.0f, 0.0f);
}

__global__ void rand_init(curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        curand_init(1989ull, 0ull, 0ull, rand_state);
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

__global__ void render(uint8_t *frame_buffer, const int image_width, const int image_height, const int num_sample, Camera **cam, Hittable **world, curandState *rand_state)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= image_width || j >= image_height)
        return;

    const int pixel_idx = (image_height - 1) * image_width - j * image_width + i;
    curandState local_rand_state = rand_state[pixel_idx];
    Color color(0.0f, 0.0f, 0.0f);
    // anti-aliasing: take the average of all the samples
    for (int s = 0; s < num_sample; ++s)
    {
        const float u = (static_cast<float>(i) + curand_uniform(&local_rand_state)) / static_cast<float>(image_width);
        const float v = (static_cast<float>(j) + curand_uniform(&local_rand_state)) / static_cast<float>(image_height);
        Ray r = (*cam)->getRay(u, v, &local_rand_state);
        color += rayColor(r, world, max_depth, &local_rand_state);
    }
    rand_state[pixel_idx] = local_rand_state;
    color /= static_cast<float>(num_sample); // take averrage, and do Gamma-2 correction
    const int ir = static_cast<uint8_t>(sqrt(color[0]) * 255.99f);
    const int ig = static_cast<uint8_t>(sqrt(color[1]) * 255.99f);
    const int ib = static_cast<uint8_t>(sqrt(color[2]) * 255.99f);

    frame_buffer[pixel_idx * 4 + 0] = ir;
    frame_buffer[pixel_idx * 4 + 1] = ig;
    frame_buffer[pixel_idx * 4 + 2] = ib;
    frame_buffer[pixel_idx * 4 + 3] = 255;
}

void displayProgress(const std::string &message, std::atomic_bool &finish_signal)
{
    finish_signal = false;
    while (!finish_signal)
    {
        std::cout << "\rGPU is " + message + " \\" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::cout << "\rGPU is " + message + " |" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::cout << "\rGPU is " + message + " /" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::cout << "\rGPU is " + message + " -" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::cout << "\rGPU finished " + message << std::endl;
}

int main()
{
    // Display information
    std::thread wait_message;
    clock_t start, stop;
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    std::cerr << "========================================================================\nNVIDIA GPU Information\n========================================================================\n";
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cerr << "Device Number: " << i << "\n"
                  << "    Device Name: " << prop.name << "\n"
                  << "    Memory Clock Rate (KHz): " << prop.memoryClockRate << "\n"
                  << "    Memory Bus Width (bits): " << prop.memoryBusWidth << "\n"
                  << "    Peak Memory Bandwidth (GB/s): " << 2.0f * static_cast<float>(prop.memoryClockRate) * (static_cast<float>(prop.memoryBusWidth) / 8.0f) / 1.0e6 << "\n\n";
    }

    // GPU random state preparation
    curandState *rand_state_world;
    curandState *rand_state_render;
    checkCudaErrors(cudaMalloc((void **)&rand_state_world, sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&rand_state_render, num_pixel * sizeof(curandState)));
    wait_message = std::thread(displayProgress, "preparing for random scene generation", std::ref(finish_signal));
    rand_init<<<1, 1>>>(rand_state_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    finish_signal = true;
    wait_message.join();

    // generate the world on the GPU
    Hittable **obj_list;
    Hittable **world;
    Camera **cam;
    constexpr size_t num_hittables = 22u * 22u + 1u + 3u;
    checkCudaErrors(cudaMalloc((void **)&obj_list, num_hittables * sizeof(Hittable *)));
    checkCudaErrors(cudaMalloc((void **)&world, sizeof(Hittable *)));
    checkCudaErrors(cudaMalloc((void **)&cam, sizeof(Camera *)));
    wait_message = std::thread(displayProgress, "generating the world", std::ref(finish_signal));
    randomScene<<<1, 1>>>(obj_list, world, cam, image_width, image_height, rand_state_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    finish_signal = true;
    wait_message.join();

    // GPU frame buffer
    uint8_t *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, frame_buffer_size));

    // launch the rendering kernel
    dim3 dim_grid(image_width / num_thread_x + 1, image_height / num_thread_y + 1, 1);
    dim3 dim_block(num_thread_x, num_thread_y, 1);
    curandState *rand_state;
    wait_message = std::thread(displayProgress, "preparing for rendering", std::ref(finish_signal));
    checkCudaErrors(cudaMalloc((void **)&rand_state, num_pixel * sizeof(curandState)));
    renderInit<<<dim_grid, dim_block>>>(rand_state, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    finish_signal = true;
    wait_message.join();

    // run the render kernel
    wait_message = std::thread(displayProgress, "rendering", std::ref(finish_signal));
    start = clock();
    render<<<dim_grid, dim_block>>>(frame_buffer, image_width, image_height, num_sample_per_pixel, cam, world, rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    finish_signal = true;
    wait_message.join();
    const float timer = (static_cast<float>(stop - start)) / static_cast<float>(CLOCKS_PER_SEC);
    std::cerr << "========================================================================\ntook GPU " << timer << " seconds to render.\n========================================================================\n";

    // save as a png file
    lodepng_encode32_file("output.png", frame_buffer, image_width, image_height);

    // free up GPU memory
    checkCudaErrors(cudaDeviceSynchronize());
    wait_message = std::thread(displayProgress, "cleaning up resources", std::ref(finish_signal));
    worldclear<<<1, 1>>>(obj_list, world, cam);
    checkCudaErrors(cudaGetLastError());
    finish_signal = true;
    wait_message.join();

    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(obj_list));
    checkCudaErrors(cudaFree(rand_state));
    checkCudaErrors(cudaFree(frame_buffer));
    cudaDeviceReset();
}
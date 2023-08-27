#pragma once

#include "Util.cuh"
#include "Perlin.cuh"

class Texture
{
public:
    //__device__ virtual ~Texture() {}

    __device__ virtual Color value(float u, float v, const Point3D &p) const = 0;
};

class SolidColor : public Texture
{
    Color color_value;

public:
    __device__ SolidColor(Color color) : color_value(color) {}

    __device__ SolidColor(float red, float green, float blue) : color_value(Color(red, green, blue)) {}

    __device__ Color value(float u, float v, const Point3D &p) const override
    {
        return color_value;
    }
};

class CheckerTexture : public Texture
{
    float inv_scale;
    Texture *even;
    Texture *odd;

public:
    __device__ CheckerTexture(float scale, Texture *even, Texture *odd) : inv_scale(1.0f / scale), even(even), odd(odd) {}

    __device__ CheckerTexture(float scale, Color even, Color odd) : inv_scale(1.0f / scale), even(new SolidColor(even)), odd(new SolidColor(odd)) {}

    //__device__ ~CheckerTexture() overrid

    __device__ Color value(float u, float v, const Point3D &p) const override
    {
        float sines = sinf(inv_scale * p.x) * sinf(inv_scale * p.y) * sinf(inv_scale * p.z);
        if (sines < 0.0f)
        {
            return odd->value(u, v, p);
        }
        else
        {
            return even->value(u, v, p);
        }
    }
};

class ImageTexture : public Texture
{
    uint8_t *image;
    int width;
    int height;
    int bytes_per_scanline;
    int bytes_per_pixel;

public:
    __device__ ImageTexture(uint8_t *image, int width, int height) : image(image), width(width), height(height), bytes_per_scanline(bytes_per_scanline), bytes_per_pixel(bytes_per_pixel) {}

    __device__ uint8_t *pixelData(int x, int y) const
    {
        // Return the address of the three bytes of the pixel at x,y (or magenta if no data).
        static unsigned char magenta[] = {255, 0, 255};
        if (image == nullptr)
            return magenta;

        x = clamp(x, 0, width);
        y = clamp(y, 0, height);

        return image + y * (3 * width) + x * 3;
    }

    __device__ Color value(float u, float v, const Point3D &p) const override
    {
        // if we have no texture data, then return solid cyan as a debugging aid
        if (height <= 0)
        {
            return Color(0.0f, 1.0f, 1.0f);
        }

        u = clamp(u, 0.0f, 1.0f);
        v = 1.0f - clamp(v, 0.0f, 1.0f);

        int i = static_cast<int>(u * width);
        int j = static_cast<int>(v * height);
        const uint8_t *pixel = pixelData(i, j);

        const float color_scale = 1.0f / 255.0f;
        return Color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
    }
};

class NoiseTexture : public Texture
{
    Perlin noise;
    float scale;

public:
    __device__ NoiseTexture(curandState *rand_state) : noise(rand_state) {}

    __device__ NoiseTexture(float scale, curandState *rand_state) : scale(scale), noise(rand_state) {}

    __device__ Color value(float u, float v, const Point3D &p) const override
    {
        Vec3 s = scale * p;
        return Color(1.0f, 1.0f, 1.0f) * 0.5f * (1.0f + sinf(s.z + 10.0f * noise.turb(s)));
    }
};
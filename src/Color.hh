#pragma once

#include "Vec3.hh"

struct pixelRGB
{
    int R;
    int G;
    int B;

    pixelRGB(int R, int G, int B) : R(R), G(G), B(B) {}
    pixelRGB() = default;
};

void writeColor(std::ostream &out, pixelRGB *canvas, int x, int y, int image_width, int image_height, Color pixel_color, int samples_per_pixel)
{
    double r = pixel_color.x;
    double g = pixel_color.y;
    double b = pixel_color.z;

    // divide the color by the number of samples and gamma-correct for gamma = 2.0
    double scale = 1.0 / samples_per_pixel;
    r = sqrt(scale * r), g = sqrt(scale * g), b = sqrt(scale * b);

    // write the translated [0, 255] value of each color component
    out << static_cast<int>(255.999 * clamp(r, 0.0, 0.999)) << " "
        << static_cast<int>(255.999 * clamp(g, 0.0, 0.999)) << " "
        << static_cast<int>(255.999 * clamp(b, 0.0, 0.999)) << std::endl;

    //  canvas[x * image_width + y] = {static_cast<int>(255.999 * clamp(r, 0.0, 0.999)), static_cast<int>(255.999 * clamp(g, 0.0, 0.999)), static_cast<int>(255.999 * clamp(b, 0.0, 0.999))};
}
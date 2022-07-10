#include <math.h>
#include <float.h>
#include "tetrahedral_kernel.h"

template <typename scalar_t>
__global__ void TetrahedralForward(const int nthreads, const scalar_t *lut, const scalar_t *image, scalar_t *output, const int dim, const int shift, const float binsize, const int width, const int height, const int batch)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {

        int r_index = index;
        int g_index = index + width * height;
        int b_index = index + width * height * 2;

        scalar_t r = image[r_index];
        scalar_t g = image[g_index];
        scalar_t b = image[b_index];

        scalar_t r_loc = r * (dim - 1);
        scalar_t g_loc = g * (dim - 1);
        scalar_t b_loc = b * (dim - 1);

        int r_0 = floor(r_loc);
        int g_0 = floor(g_loc);
        int b_0 = floor(b_loc);
        int r_1 = r_0 + 1;
        int g_1 = g_0 + 1;
        int b_1 = b_0 + 1;

        r_0 = CLIP(r_0, 0, dim - 1);
        g_0 = CLIP(g_0, 0, dim - 1);
        b_0 = CLIP(b_0, 0, dim - 1);
        r_1 = CLIP(r_1, 0, dim - 1);
        g_1 = CLIP(g_1, 0, dim - 1);
        b_1 = CLIP(b_1, 0, dim - 1);

        scalar_t r_d = r_loc - r_0;
        scalar_t g_d = g_loc - g_0;
        scalar_t b_d = b_loc - b_0;

        scalar_t w000 = (1 - r_d) * (1 - g_d) * (1 - b_d);
        scalar_t w100 = r_d * (1 - g_d) * (1 - b_d);
        scalar_t w010 = (1 - r_d) * g_d * (1 - b_d);
        scalar_t w110 = r_d * g_d * (1 - b_d);
        scalar_t w001 = (1 - r_d) * (1 - g_d) * b_d;
        scalar_t w101 = r_d * (1 - g_d) * b_d;
        scalar_t w011 = (1 - r_d) * g_d * b_d;
        scalar_t w111 = r_d * g_d * b_d;

        int id000 = INDEX(0, r_0, g_0, b_0, dim, dim, dim);
        int id100 = INDEX(0, r_1, g_0, b_0, dim, dim, dim);
        int id010 = INDEX(0, r_0, g_1, b_0, dim, dim, dim);
        int id110 = INDEX(0, r_1, g_1, b_0, dim, dim, dim);
        int id001 = INDEX(0, r_0, g_0, b_1, dim, dim, dim);
        int id101 = INDEX(0, r_1, g_0, b_1, dim, dim, dim);
        int id011 = INDEX(0, r_0, g_1, b_1, dim, dim, dim);
        int id111 = INDEX(0, r_1, g_1, b_1, dim, dim, dim);

        // compute value based on 6 cases
        if (r_d > g_d && g_d > b_d)
        {
            output[r_index] = (1 - r_d) * lut[INDEX(0, r_0, g_0, b_0, dim, dim, dim)] +
                              (r_d - g_d) * lut[INDEX(0, r_1, g_0, b_0, dim, dim, dim)] +
                              (g_d - b_d) * lut[INDEX(0, r_1, g_1, b_0, dim, dim, dim)] +
                              b_d * lut[INDEX(0, r_1, g_1, b_1, dim, dim, dim)];

            output[g_index] = (1 - r_d) * lut[INDEX(1, r_0, g_0, b_0, dim, dim, dim)] +
                              (r_d - g_d) * lut[INDEX(1, r_1, g_0, b_0, dim, dim, dim)] +
                              (g_d - b_d) * lut[INDEX(1, r_1, g_1, b_0, dim, dim, dim)] +
                              b_d * lut[INDEX(1, r_1, g_1, b_1, dim, dim, dim)];

            output[b_index] = (1 - r_d) * lut[INDEX(2, r_0, g_0, b_0, dim, dim, dim)] +
                              (r_d - g_d) * lut[INDEX(2, r_1, g_0, b_0, dim, dim, dim)] +
                              (g_d - b_d) * lut[INDEX(2, r_1, g_1, b_0, dim, dim, dim)] +
                              b_d * lut[INDEX(2, r_1, g_1, b_1, dim, dim, dim)];
        }
        else if (r_d > g_d && r_d > b_d)
        {
            output[r_index] = (1 - r_d) * lut[INDEX(0, r_0, g_0, b_0, dim, dim, dim)] +
                              (r_d - b_d) * lut[INDEX(0, r_1, g_0, b_0, dim, dim, dim)] +
                              (b_d - g_d) * lut[INDEX(0, r_1, g_0, b_1, dim, dim, dim)] +
                              g_d * lut[INDEX(0, r_1, g_1, b_1, dim, dim, dim)];

            output[g_index] = (1 - r_d) * lut[INDEX(1, r_0, g_0, b_0, dim, dim, dim)] +
                              (r_d - b_d) * lut[INDEX(1, r_1, g_0, b_0, dim, dim, dim)] +
                              (b_d - g_d) * lut[INDEX(1, r_1, g_0, b_1, dim, dim, dim)] +
                              g_d * lut[INDEX(1, r_1, g_1, b_1, dim, dim, dim)];

            output[b_index] = (1 - r_d) * lut[INDEX(2, r_0, g_0, b_0, dim, dim, dim)] +
                              (r_d - b_d) * lut[INDEX(2, r_1, g_0, b_0, dim, dim, dim)] +
                              (b_d - g_d) * lut[INDEX(2, r_1, g_0, b_1, dim, dim, dim)] +
                              g_d * lut[INDEX(2, r_1, g_1, b_1, dim, dim, dim)];
        }
        else if (r_d > g_d && g_d <= b_d && r_d <= b_d)
        {
            output[r_index] = (1 - b_d) * lut[INDEX(0, r_0, g_0, b_0, dim, dim, dim)] +
                              (b_d - r_d) * lut[INDEX(0, r_0, g_0, b_1, dim, dim, dim)] +
                              (r_d - g_d) * lut[INDEX(0, r_1, g_0, b_1, dim, dim, dim)] +
                              g_d * lut[INDEX(0, r_1, g_1, b_1, dim, dim, dim)];

            output[g_index] = (1 - b_d) * lut[INDEX(1, r_0, g_0, b_0, dim, dim, dim)] +
                              (b_d - r_d) * lut[INDEX(1, r_0, g_0, b_1, dim, dim, dim)] +
                              (r_d - g_d) * lut[INDEX(1, r_1, g_0, b_1, dim, dim, dim)] +
                              g_d * lut[INDEX(1, r_1, g_1, b_1, dim, dim, dim)];

            output[b_index] = (1 - b_d) * lut[INDEX(2, r_0, g_0, b_0, dim, dim, dim)] +
                              (b_d - r_d) * lut[INDEX(2, r_0, g_0, b_1, dim, dim, dim)] +
                              (r_d - g_d) * lut[INDEX(2, r_1, g_0, b_1, dim, dim, dim)] +
                              g_d * lut[INDEX(2, r_1, g_1, b_1, dim, dim, dim)];
        }
        else if (r_d <= g_d && b_d > g_d)
        {
            output[r_index] = (1 - b_d) * lut[INDEX(0, r_0, g_0, b_0, dim, dim, dim)] +
                              (b_d - g_d) * lut[INDEX(0, r_0, g_0, b_1, dim, dim, dim)] +
                              (g_d - r_d) * lut[INDEX(0, r_0, g_1, b_1, dim, dim, dim)] +
                              r_d * lut[INDEX(0, r_1, g_1, b_1, dim, dim, dim)];

            output[g_index] = (1 - b_d) * lut[INDEX(1, r_0, g_0, b_0, dim, dim, dim)] +
                              (b_d - g_d) * lut[INDEX(1, r_0, g_0, b_1, dim, dim, dim)] +
                              (g_d - r_d) * lut[INDEX(1, r_0, g_1, b_1, dim, dim, dim)] +
                              r_d * lut[INDEX(1, r_1, g_1, b_1, dim, dim, dim)];

            output[b_index] = (1 - b_d) * lut[INDEX(2, r_0, g_0, b_0, dim, dim, dim)] +
                              (b_d - g_d) * lut[INDEX(2, r_0, g_0, b_1, dim, dim, dim)] +
                              (g_d - r_d) * lut[INDEX(2, r_0, g_1, b_1, dim, dim, dim)] +
                              r_d * lut[INDEX(2, r_1, g_1, b_1, dim, dim, dim)];
        }
        else if (r_d <= g_d && b_d > r_d)
        {
            output[r_index] = (1 - g_d) * lut[INDEX(0, r_0, g_0, b_0, dim, dim, dim)] +
                              (g_d - b_d) * lut[INDEX(0, r_0, g_1, b_0, dim, dim, dim)] +
                              (b_d - r_d) * lut[INDEX(0, r_0, g_1, b_1, dim, dim, dim)] +
                              r_d * lut[INDEX(0, r_1, g_1, b_1, dim, dim, dim)];

            output[g_index] = (1 - g_d) * lut[INDEX(1, r_0, g_0, b_0, dim, dim, dim)] +
                              (g_d - b_d) * lut[INDEX(1, r_0, g_1, b_0, dim, dim, dim)] +
                              (b_d - r_d) * lut[INDEX(1, r_0, g_1, b_1, dim, dim, dim)] +
                              r_d * lut[INDEX(1, r_1, g_1, b_1, dim, dim, dim)];

            output[b_index] = (1 - g_d) * lut[INDEX(2, r_0, g_0, b_0, dim, dim, dim)] +
                              (g_d - b_d) * lut[INDEX(2, r_0, g_1, b_0, dim, dim, dim)] +
                              (b_d - r_d) * lut[INDEX(2, r_0, g_1, b_1, dim, dim, dim)] +
                              r_d * lut[INDEX(2, r_1, g_1, b_1, dim, dim, dim)];
        }
        else
        {
            output[r_index] = (1 - g_d) * lut[INDEX(0, r_0, g_0, b_0, dim, dim, dim)] +
                              (g_d - r_d) * lut[INDEX(0, r_0, g_1, b_0, dim, dim, dim)] +
                              (r_d - b_d) * lut[INDEX(0, r_1, g_1, b_0, dim, dim, dim)] +
                              b_d * lut[INDEX(0, r_1, g_1, b_1, dim, dim, dim)];

            output[g_index] = (1 - g_d) * lut[INDEX(1, r_0, g_0, b_0, dim, dim, dim)] +
                              (g_d - r_d) * lut[INDEX(1, r_0, g_1, b_0, dim, dim, dim)] +
                              (r_d - b_d) * lut[INDEX(1, r_1, g_1, b_0, dim, dim, dim)] +
                              b_d * lut[INDEX(1, r_1, g_1, b_1, dim, dim, dim)];

            output[b_index] = (1 - g_d) * lut[INDEX(2, r_0, g_0, b_0, dim, dim, dim)] +
                              (g_d - r_d) * lut[INDEX(2, r_0, g_1, b_0, dim, dim, dim)] +
                              (r_d - b_d) * lut[INDEX(2, r_1, g_1, b_0, dim, dim, dim)] +
                              b_d * lut[INDEX(2, r_1, g_1, b_1, dim, dim, dim)];
        }
    }
}

template <typename scalar_t>
__global__ void TetrahedralBackward(const int nthreads, const scalar_t *image, const scalar_t *image_grad, scalar_t *lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int batch)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {

        int r_index = index;
        int g_index = index + width * height;
        int b_index = index + width * height * 2;

        scalar_t r = image[r_index];
        scalar_t g = image[g_index];
        scalar_t b = image[b_index];

        scalar_t r_loc = r * (dim - 1);
        scalar_t g_loc = g * (dim - 1);
        scalar_t b_loc = b * (dim - 1);

        int r_0 = floor(r_loc);
        int g_0 = floor(g_loc);
        int b_0 = floor(b_loc);
        int r_1 = r_0 + 1;
        int g_1 = g_0 + 1;
        int b_1 = b_0 + 1;

        r_0 = CLIP(r_0, 0, dim - 1);
        g_0 = CLIP(g_0, 0, dim - 1);
        b_0 = CLIP(b_0, 0, dim - 1);
        r_1 = CLIP(r_1, 0, dim - 1);
        g_1 = CLIP(g_1, 0, dim - 1);
        b_1 = CLIP(b_1, 0, dim - 1);

        scalar_t r_d = r_loc - r_0;
        scalar_t g_d = g_loc - g_0;
        scalar_t b_d = b_loc - b_0;

        scalar_t w000 = (1 - r_d) * (1 - g_d) * (1 - b_d);
        scalar_t w100 = r_d * (1 - g_d) * (1 - b_d);
        scalar_t w010 = (1 - r_d) * g_d * (1 - b_d);
        scalar_t w110 = r_d * g_d * (1 - b_d);
        scalar_t w001 = (1 - r_d) * (1 - g_d) * b_d;
        scalar_t w101 = r_d * (1 - g_d) * b_d;
        scalar_t w011 = (1 - r_d) * g_d * b_d;
        scalar_t w111 = r_d * g_d * b_d;

        int id000 = INDEX(0, r_0, g_0, b_0, dim, dim, dim);
        int id100 = INDEX(0, r_1, g_0, b_0, dim, dim, dim);
        int id010 = INDEX(0, r_0, g_1, b_0, dim, dim, dim);
        int id110 = INDEX(0, r_1, g_1, b_0, dim, dim, dim);
        int id001 = INDEX(0, r_0, g_0, b_1, dim, dim, dim);
        int id101 = INDEX(0, r_1, g_0, b_1, dim, dim, dim);
        int id011 = INDEX(0, r_0, g_1, b_1, dim, dim, dim);
        int id111 = INDEX(0, r_1, g_1, b_1, dim, dim, dim);

        // compute gradient based on 6 cases
        if (r_d > g_d && g_d > b_d)
        {
            atomicAdd(lut_grad + id000, image_grad[r_index] * (1 - r_d));
            atomicAdd(lut_grad + id100, image_grad[r_index] * (r_d - g_d));
            atomicAdd(lut_grad + id110, image_grad[r_index] * (g_d - b_d));
            atomicAdd(lut_grad + id111, image_grad[r_index] * b_d);

            atomicAdd(lut_grad + id000 + shift, image_grad[g_index] * (1 - r_d));
            atomicAdd(lut_grad + id100 + shift, image_grad[g_index] * (r_d - g_d));
            atomicAdd(lut_grad + id110 + shift, image_grad[g_index] * (g_d - b_d));
            atomicAdd(lut_grad + id111 + shift, image_grad[g_index] * b_d);

            atomicAdd(lut_grad + id000 + shift * 2, image_grad[b_index] * (1 - r_d));
            atomicAdd(lut_grad + id100 + shift * 2, image_grad[b_index] * (r_d - g_d));
            atomicAdd(lut_grad + id110 + shift * 2, image_grad[b_index] * (g_d - b_d));
            atomicAdd(lut_grad + id111 + shift * 2, image_grad[b_index] * b_d);
        }
        else if (r_d > g_d && r_d > b_d)
        {
            atomicAdd(lut_grad + id000, image_grad[r_index] * (1 - r_d));
            atomicAdd(lut_grad + id100, image_grad[r_index] * (r_d - b_d));
            atomicAdd(lut_grad + id101, image_grad[r_index] * (b_d - g_d));
            atomicAdd(lut_grad + id111, image_grad[r_index] * g_d);

            atomicAdd(lut_grad + id000 + shift, image_grad[g_index] * (1 - r_d));
            atomicAdd(lut_grad + id100 + shift, image_grad[g_index] * (r_d - b_d));
            atomicAdd(lut_grad + id101 + shift, image_grad[g_index] * (b_d - g_d));
            atomicAdd(lut_grad + id111 + shift, image_grad[g_index] * g_d);

            atomicAdd(lut_grad + id000 + shift * 2, image_grad[b_index] * (1 - r_d));
            atomicAdd(lut_grad + id100 + shift * 2, image_grad[b_index] * (r_d - b_d));
            atomicAdd(lut_grad + id101 + shift * 2, image_grad[b_index] * (b_d - g_d));
            atomicAdd(lut_grad + id111 + shift * 2, image_grad[b_index] * g_d);
        }
        else if (r_d > g_d && g_d <= b_d && r_d <= b_d)
        {
            atomicAdd(lut_grad + id000, image_grad[r_index] * (1 - b_d));
            atomicAdd(lut_grad + id001, image_grad[r_index] * (b_d - r_d));
            atomicAdd(lut_grad + id101, image_grad[r_index] * (r_d - g_d));
            atomicAdd(lut_grad + id111, image_grad[r_index] * g_d);

            atomicAdd(lut_grad + id000 + shift, image_grad[g_index] * (1 - b_d));
            atomicAdd(lut_grad + id001 + shift, image_grad[g_index] * (b_d - r_d));
            atomicAdd(lut_grad + id101 + shift, image_grad[g_index] * (r_d - g_d));
            atomicAdd(lut_grad + id111 + shift, image_grad[g_index] * g_d);

            atomicAdd(lut_grad + id000 + shift * 2, image_grad[b_index] * (1 - b_d));
            atomicAdd(lut_grad + id001 + shift * 2, image_grad[b_index] * (b_d - r_d));
            atomicAdd(lut_grad + id101 + shift * 2, image_grad[b_index] * (r_d - g_d));
            atomicAdd(lut_grad + id111 + shift * 2, image_grad[b_index] * g_d);
        }
        else if (r_d <= g_d && b_d > g_d)
        {
            atomicAdd(lut_grad + id000, image_grad[r_index] * (1 - b_d));
            atomicAdd(lut_grad + id001, image_grad[r_index] * (b_d - g_d));
            atomicAdd(lut_grad + id011, image_grad[r_index] * (g_d - r_d));
            atomicAdd(lut_grad + id111, image_grad[r_index] * r_d);

            atomicAdd(lut_grad + id000 + shift, image_grad[g_index] * (1 - b_d));
            atomicAdd(lut_grad + id001 + shift, image_grad[g_index] * (b_d - g_d));
            atomicAdd(lut_grad + id011 + shift, image_grad[g_index] * (g_d - r_d));
            atomicAdd(lut_grad + id111 + shift, image_grad[g_index] * r_d);

            atomicAdd(lut_grad + id000 + shift * 2, image_grad[b_index] * (1 - b_d));
            atomicAdd(lut_grad + id001 + shift * 2, image_grad[b_index] * (b_d - g_d));
            atomicAdd(lut_grad + id011 + shift * 2, image_grad[b_index] * (g_d - r_d));
            atomicAdd(lut_grad + id111 + shift * 2, image_grad[b_index] * r_d);
        }
        else if (r_d <= g_d && b_d > r_d)
        {
            atomicAdd(lut_grad + id000, image_grad[r_index] * (1 - g_d));
            atomicAdd(lut_grad + id010, image_grad[r_index] * (g_d - b_d));
            atomicAdd(lut_grad + id011, image_grad[r_index] * (b_d - r_d));
            atomicAdd(lut_grad + id111, image_grad[r_index] * r_d);

            atomicAdd(lut_grad + id000 + shift, image_grad[g_index] * (1 - g_d));
            atomicAdd(lut_grad + id010 + shift, image_grad[g_index] * (g_d - b_d));
            atomicAdd(lut_grad + id011 + shift, image_grad[g_index] * (b_d - r_d));
            atomicAdd(lut_grad + id111 + shift, image_grad[g_index] * r_d);

            atomicAdd(lut_grad + id000 + shift * 2, image_grad[b_index] * (1 - g_d));
            atomicAdd(lut_grad + id010 + shift * 2, image_grad[b_index] * (g_d - b_d));
            atomicAdd(lut_grad + id011 + shift * 2, image_grad[b_index] * (b_d - r_d));
            atomicAdd(lut_grad + id111 + shift * 2, image_grad[b_index] * r_d);
        }
        else
        {
            atomicAdd(lut_grad + id000, image_grad[r_index] * (1 - g_d));
            atomicAdd(lut_grad + id010, image_grad[r_index] * (g_d - r_d));
            atomicAdd(lut_grad + id110, image_grad[r_index] * (r_d - b_d));
            atomicAdd(lut_grad + id111, image_grad[r_index] * b_d);

            atomicAdd(lut_grad + id000 + shift, image_grad[g_index] * (1 - g_d));
            atomicAdd(lut_grad + id010 + shift, image_grad[g_index] * (g_d - r_d));
            atomicAdd(lut_grad + id110 + shift, image_grad[g_index] * (r_d - b_d));
            atomicAdd(lut_grad + id111 + shift, image_grad[g_index] * b_d);

            atomicAdd(lut_grad + id000 + shift * 2, image_grad[b_index] * (1 - g_d));
            atomicAdd(lut_grad + id010 + shift * 2, image_grad[b_index] * (g_d - r_d));
            atomicAdd(lut_grad + id110 + shift * 2, image_grad[b_index] * (r_d - b_d));
            atomicAdd(lut_grad + id111 + shift * 2, image_grad[b_index] * b_d);
        }
    }
}
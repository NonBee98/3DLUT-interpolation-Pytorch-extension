#include <math.h>
#include <float.h>
#include "trilinear_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

#define CLIP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define INDEX(a, b, c, d, d1, d2, d3) ((a) * (d1) * (d2) * (d3) + (b) * (d2) * (d3) + (c) * (d3) + (d))

template <typename scalar_t>
__global__ void TriLinearForward(const int nthreads, const scalar_t *lut, const scalar_t *image, scalar_t *output, const int dim, const int shift, const float binsize, const int width, const int height, const int batch)
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

        output[r_index] = w000 * lut[id000] + w100 * lut[id100] +
                          w010 * lut[id010] + w110 * lut[id110] +
                          w001 * lut[id001] + w101 * lut[id101] +
                          w011 * lut[id011] + w111 * lut[id111];

        output[g_index] = w000 * lut[id000 + shift] + w100 * lut[id100 + shift] +
                          w010 * lut[id010 + shift] + w110 * lut[id110 + shift] +
                          w001 * lut[id001 + shift] + w101 * lut[id101 + shift] +
                          w011 * lut[id011 + shift] + w111 * lut[id111 + shift];

        output[b_index] = w000 * lut[id000 + shift * 2] + w100 * lut[id100 + shift * 2] +
                          w010 * lut[id010 + shift * 2] + w110 * lut[id110 + shift * 2] +
                          w001 * lut[id001 + shift * 2] + w101 * lut[id101 + shift * 2] +
                          w011 * lut[id011 + shift * 2] + w111 * lut[id111 + shift * 2];
    }
}

template <typename scalar_t>
__global__ void TriLinearBackward(const int nthreads, const scalar_t *image, const scalar_t *image_grad, scalar_t *lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int batch)
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

        atomicAdd(lut_grad + id000, image_grad[r_index] * w000);
        atomicAdd(lut_grad + id100, image_grad[r_index] * w100);
        atomicAdd(lut_grad + id010, image_grad[r_index] * w010);
        atomicAdd(lut_grad + id110, image_grad[r_index] * w110);
        atomicAdd(lut_grad + id001, image_grad[r_index] * w001);
        atomicAdd(lut_grad + id101, image_grad[r_index] * w101);
        atomicAdd(lut_grad + id011, image_grad[r_index] * w011);
        atomicAdd(lut_grad + id111, image_grad[r_index] * w111);

        atomicAdd(lut_grad + id000 + shift, image_grad[g_index] * w000);
        atomicAdd(lut_grad + id100 + shift, image_grad[g_index] * w100);
        atomicAdd(lut_grad + id010 + shift, image_grad[g_index] * w010);
        atomicAdd(lut_grad + id110 + shift, image_grad[g_index] * w110);
        atomicAdd(lut_grad + id001 + shift, image_grad[g_index] * w001);
        atomicAdd(lut_grad + id101 + shift, image_grad[g_index] * w101);
        atomicAdd(lut_grad + id011 + shift, image_grad[g_index] * w011);
        atomicAdd(lut_grad + id111 + shift, image_grad[g_index] * w111);

        atomicAdd(lut_grad + id000 + shift * 2, image_grad[b_index] * w000);
        atomicAdd(lut_grad + id100 + shift * 2, image_grad[b_index] * w100);
        atomicAdd(lut_grad + id010 + shift * 2, image_grad[b_index] * w010);
        atomicAdd(lut_grad + id110 + shift * 2, image_grad[b_index] * w110);
        atomicAdd(lut_grad + id001 + shift * 2, image_grad[b_index] * w001);
        atomicAdd(lut_grad + id101 + shift * 2, image_grad[b_index] * w101);
        atomicAdd(lut_grad + id011 + shift * 2, image_grad[b_index] * w011);
        atomicAdd(lut_grad + id111 + shift * 2, image_grad[b_index] * w111);
    }
}

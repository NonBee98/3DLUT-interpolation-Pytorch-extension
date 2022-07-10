#ifndef _TETRAHEDRAL_KERNEL
#define _TETRAHEDRAL_KERNEL
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CLIP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define INDEX(a, b, c, d, d1, d2, d3) ((a) * (d1) * (d2) * (d3) + (b) * (d2) * (d3) + (c) * (d3) + (d))
#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void TetrahedralForward(const int nthreads, const scalar_t *lut, const scalar_t *image, scalar_t *output, const int dim, const int shift, const float binsize, const int width, const int height, const int batch);

template <typename scalar_t>
__global__ void TetrahedralBackward(const int nthreads, const scalar_t *image, const scalar_t *image_grad, scalar_t *lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int batch);

#endif

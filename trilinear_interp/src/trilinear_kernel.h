#ifndef _TRILINEAR_KERNEL
#define _TRILINEAR_KERNEL

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void TriLinearForward(const int nthreads, const scalar_t *lut, const scalar_t *image, scalar_t *output, const int dim, const int shift, const float binsize, const int width, const int height, const int batch);

template <typename scalar_t>
__global__ void TriLinearBackward(const int nthreads, const scalar_t *image, const scalar_t *image_grad, scalar_t *lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int batch);

#endif

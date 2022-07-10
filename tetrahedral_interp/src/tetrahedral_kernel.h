#ifndef _TETRAHEDRAL_KERNEL
#define _TETRAHEDRAL_KERNEL
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void TetrahedralForward(const int nthreads, const float *lut, const float *image, float *output, const int dim, const int shift, const float binsize, const int width, const int height, const int batch);

int TetrahedralForwardLaucher(const float *lut, const float *image, float *output, const int lut_dim, const int shift, const float binsize, const int width, const int height, const int batch, cudaStream_t stream);

__global__ void TetrahedralBackward(const int nthreads, const float *image, const float *image_grad, float *lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int batch);

int TetrahedralBackwardLaucher(const float *image, const float *image_grad, float *lut_grad, const int lut_dim, const int shift, const float binsize, const int width, const int height, const int batch, cudaStream_t stream);

#endif

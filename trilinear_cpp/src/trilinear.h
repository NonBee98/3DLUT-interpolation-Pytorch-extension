#ifndef TRILINEAR_H
#define TRILINEAR_H

#include <torch/extension.h>

#define CLIP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define INDEX(a, b, c, d, d1, d2, d3) ((a) * (d1) * (d2) * (d3) + (b) * (d2) * (d3) + (c) * (d3) + (d))

int trilinear_forward(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                      int lut_dim, int shift, float binsize, int width, int height, int batch);

int trilinear_backward(torch::Tensor image, torch::Tensor image_grad, torch::Tensor lut_grad,
                       int lut_dim, int shift, float binsize, int width, int height, int batch);

template <typename scalar_t>
void TriLinearForwardCpu(const scalar_t *lut, const scalar_t *image, scalar_t *output, const int dim, const int shift, const float binsize, const int width, const int height, const int channels, const int batch);

template <typename scalar_t>
void TriLinearBackwardCpu(const scalar_t *image, const scalar_t *image_grad, scalar_t *lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int channels, const int batch);

#endif

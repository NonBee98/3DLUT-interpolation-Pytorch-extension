#ifndef TETRAHEDRAL_H
#define TETRAHEDRAL_H

#include <torch/extension.h>

int tetrahedral_forward(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                        int lut_dim, int shift, float binsize, int width, int height, int batch);

int tetrahedral_backward(torch::Tensor image, torch::Tensor image_grad, torch::Tensor lut_grad,
                         int lut_dim, int shift, float binsize, int width, int height, int batch);

#endif

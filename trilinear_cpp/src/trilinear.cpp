#include "trilinear.h"

int trilinear_forward(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                      int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    auto image_size = image.sizes();
    int channels = image_size[1];

    AT_DISPATCH_FLOATING_TYPES(lut.scalar_type(), "trilinear_forward_cpp",
                               ([&]
                                { TriLinearForwardCpu<scalar_t>(
                                      lut.data_ptr<scalar_t>(),
                                      image.data_ptr<scalar_t>(),
                                      output.data_ptr<scalar_t>(),
                                      lut_dim, shift, binsize, width,
                                      height, channels, batch); }));

    return 1;
}

int trilinear_backward(torch::Tensor image, torch::Tensor image_grad, torch::Tensor lut_grad,
                       int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    auto image_size = image.sizes();
    int channels = image_size[1];

    AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "trilinear_backward_cpp",
                               ([&]
                                { TriLinearBackwardCpu<scalar_t>(
                                      image.data_ptr<scalar_t>(),
                                      image_grad.data_ptr<scalar_t>(),
                                      lut_grad.data_ptr<scalar_t>(),
                                      lut_dim, shift, binsize, width,
                                      height, channels, batch); }));

    return 1;
}

template <typename scalar_t>
void TriLinearForwardCpu(const scalar_t *lut, const scalar_t *image, scalar_t *output, const int dim, const int shift, const float binsize, const int width, const int height, const int channels, const int batch)
{
    for (int batch_index = 0; batch_index < batch; ++batch_index)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int r_index = INDEX(batch_index, 0, h, w, 3, height, width);
                int g_index = INDEX(batch_index, 1, h, w, 3, height, width);
                int b_index = INDEX(batch_index, 2, h, w, 3, height, width);

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

                output[r_index] = w000 * lut[INDEX(0, r_0, g_0, b_0, dim, dim, dim)] +
                                  w100 * lut[INDEX(0, r_1, g_0, b_0, dim, dim, dim)] +
                                  w010 * lut[INDEX(0, r_0, g_1, b_0, dim, dim, dim)] +
                                  w110 * lut[INDEX(0, r_1, g_1, b_0, dim, dim, dim)] +
                                  w001 * lut[INDEX(0, r_0, g_0, b_1, dim, dim, dim)] +
                                  w101 * lut[INDEX(0, r_1, g_0, b_1, dim, dim, dim)] +
                                  w011 * lut[INDEX(0, r_0, g_1, b_1, dim, dim, dim)] +
                                  w111 * lut[INDEX(0, r_1, g_1, b_1, dim, dim, dim)];

                output[g_index] = w000 * lut[INDEX(1, r_0, g_0, b_0, dim, dim, dim)] +
                                  w100 * lut[INDEX(1, r_1, g_0, b_0, dim, dim, dim)] +
                                  w010 * lut[INDEX(1, r_0, g_1, b_0, dim, dim, dim)] +
                                  w110 * lut[INDEX(1, r_1, g_1, b_0, dim, dim, dim)] +
                                  w001 * lut[INDEX(1, r_0, g_0, b_1, dim, dim, dim)] +
                                  w101 * lut[INDEX(1, r_1, g_0, b_1, dim, dim, dim)] +
                                  w011 * lut[INDEX(1, r_0, g_1, b_1, dim, dim, dim)] +
                                  w111 * lut[INDEX(1, r_1, g_1, b_1, dim, dim, dim)];

                output[b_index] = w000 * lut[INDEX(2, r_0, g_0, b_0, dim, dim, dim)] +
                                  w100 * lut[INDEX(2, r_1, g_0, b_0, dim, dim, dim)] +
                                  w010 * lut[INDEX(2, r_0, g_1, b_0, dim, dim, dim)] +
                                  w110 * lut[INDEX(2, r_1, g_1, b_0, dim, dim, dim)] +
                                  w001 * lut[INDEX(2, r_0, g_0, b_1, dim, dim, dim)] +
                                  w101 * lut[INDEX(2, r_1, g_0, b_1, dim, dim, dim)] +
                                  w011 * lut[INDEX(2, r_0, g_1, b_1, dim, dim, dim)] +
                                  w111 * lut[INDEX(2, r_1, g_1, b_1, dim, dim, dim)];
            }
        }
    }
}

template <typename scalar_t>
void TriLinearBackwardCpu(const scalar_t *image, const scalar_t *image_grad, scalar_t *lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int channels, const int batch)
{
    for (int batch_index = 0; batch_index < batch; ++batch_index)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int r_index = INDEX(batch_index, 0, h, w, 3, height, width);
                int g_index = INDEX(batch_index, 1, h, w, 3, height, width);
                int b_index = INDEX(batch_index, 2, h, w, 3, height, width);

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

                lut_grad[id000] += w000 * image_grad[r_index];
                lut_grad[id100] += w100 * image_grad[r_index];
                lut_grad[id010] += w010 * image_grad[r_index];
                lut_grad[id110] += w110 * image_grad[r_index];
                lut_grad[id001] += w001 * image_grad[r_index];
                lut_grad[id101] += w101 * image_grad[r_index];
                lut_grad[id011] += w011 * image_grad[r_index];
                lut_grad[id111] += w111 * image_grad[r_index];

                lut_grad[id000 + shift] += w000 * image_grad[g_index];
                lut_grad[id100 + shift] += w100 * image_grad[g_index];
                lut_grad[id010 + shift] += w010 * image_grad[g_index];
                lut_grad[id110 + shift] += w110 * image_grad[g_index];
                lut_grad[id001 + shift] += w001 * image_grad[g_index];
                lut_grad[id101 + shift] += w101 * image_grad[g_index];
                lut_grad[id011 + shift] += w011 * image_grad[g_index];
                lut_grad[id111 + shift] += w111 * image_grad[g_index];

                lut_grad[id000 + shift * 2] += w000 * image_grad[b_index];
                lut_grad[id100 + shift * 2] += w100 * image_grad[b_index];
                lut_grad[id010 + shift * 2] += w010 * image_grad[b_index];
                lut_grad[id110 + shift * 2] += w110 * image_grad[b_index];
                lut_grad[id001 + shift * 2] += w001 * image_grad[b_index];
                lut_grad[id101 + shift * 2] += w101 * image_grad[b_index];
                lut_grad[id011 + shift * 2] += w011 * image_grad[b_index];
                lut_grad[id111 + shift * 2] += w111 * image_grad[b_index];
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &trilinear_forward, "Trilinear forward");
    m.def("backward", &trilinear_backward, "Trilinear backward");
}

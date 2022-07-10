#include "tetrahedral.h"
#define CLIP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define INDEX(a, b, c, d, d1, d2, d3) ((a) * (d1) * (d2) * (d3) + (b) * (d2) * (d3) + (c) * (d3) + (d))

template <typename scalar_t>
void TetrahedralForwardCpu(const scalar_t *lut, const scalar_t *image, scalar_t *output, const int dim, const int shift, const float binsize, const int width, const int height, const int channels, const int batch);

template <typename scalar_t>
void TetrahedralBackwardCpu(const scalar_t *image, const scalar_t *image_grad, scalar_t *lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int channels, const int batch);

int tetrahedral_forward(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                        int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    auto image_size = image.sizes();
    int channels = image_size[1];

    AT_DISPATCH_FLOATING_TYPES(lut.scalar_type(), "tetrahedral_forward_cpp",
                               ([&]
                                { TetrahedralForwardCpu<scalar_t>(
                                      lut.data_ptr<scalar_t>(),
                                      image.data_ptr<scalar_t>(),
                                      output.data_ptr<scalar_t>(),
                                      lut_dim, shift, binsize, width,
                                      height, channels, batch); }));

    return 1;
}

int tetrahedral_backward(torch::Tensor image, torch::Tensor image_grad, torch::Tensor lut_grad,
                         int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    auto image_size = image.sizes();
    int channels = image_size[1];

    AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "tetrahedral_backward_cpp",
                               ([&]
                                { TetrahedralBackwardCpu<scalar_t>(
                                      image.data_ptr<scalar_t>(),
                                      image_grad.data_ptr<scalar_t>(),
                                      lut_grad.data_ptr<scalar_t>(),
                                      lut_dim, shift, binsize, width,
                                      height, channels, batch); }));

    return 1;
}

template <typename scalar_t>
void TetrahedralForwardCpu(const scalar_t *lut, const scalar_t *image, scalar_t *output, const int dim, const int shift, const float binsize, const int width, const int height, const int channels, const int batch)
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
    }
}

template <typename scalar_t>
void TetrahedralBackwardCpu(const scalar_t *image, const scalar_t *image_grad, scalar_t *lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int channels, const int batch)
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
                    lut_grad[id000] += (1 - r_d) * image_grad[r_index];
                    lut_grad[id100] += (r_d - g_d) * image_grad[r_index];
                    lut_grad[id110] += (g_d - b_d) * image_grad[r_index];
                    lut_grad[id111] += b_d * image_grad[r_index];

                    lut_grad[id000 + shift] += (1 - r_d) * image_grad[g_index];
                    lut_grad[id100 + shift] += (r_d - g_d) * image_grad[g_index];
                    lut_grad[id110 + shift] += (g_d - b_d) * image_grad[g_index];
                    lut_grad[id111 + shift] += b_d * image_grad[g_index];

                    lut_grad[id000 + shift * 2] += (1 - r_d) * image_grad[b_index];
                    lut_grad[id100 + shift * 2] += (r_d - g_d) * image_grad[b_index];
                    lut_grad[id110 + shift * 2] += (g_d - b_d) * image_grad[b_index];
                    lut_grad[id111 + shift * 2] += b_d * image_grad[b_index];
                }
                else if (r_d > g_d && r_d > b_d)
                {
                    lut_grad[id000] += (1 - r_d) * image_grad[r_index];
                    lut_grad[id100] += (r_d - b_d) * image_grad[r_index];
                    lut_grad[id101] += (b_d - g_d) * image_grad[r_index];
                    lut_grad[id111] += g_d * image_grad[r_index];

                    lut_grad[id000 + shift] += (1 - r_d) * image_grad[g_index];
                    lut_grad[id100 + shift] += (r_d - b_d) * image_grad[g_index];
                    lut_grad[id101 + shift] += (b_d - g_d) * image_grad[g_index];
                    lut_grad[id111 + shift] += g_d * image_grad[g_index];

                    lut_grad[id000 + shift * 2] += (1 - r_d) * image_grad[b_index];
                    lut_grad[id100 + shift * 2] += (r_d - b_d) * image_grad[b_index];
                    lut_grad[id101 + shift * 2] += (b_d - g_d) * image_grad[b_index];
                    lut_grad[id111 + shift * 2] += g_d * image_grad[b_index];
                }
                else if (r_d > g_d && g_d <= b_d && r_d <= b_d)
                {
                    lut_grad[id000] += (1 - b_d) * image_grad[r_index];
                    lut_grad[id001] += (b_d - r_d) * image_grad[r_index];
                    lut_grad[id101] += (r_d - g_d) * image_grad[r_index];
                    lut_grad[id111] += g_d * image_grad[r_index];

                    lut_grad[id000 + shift] += (1 - b_d) * image_grad[g_index];
                    lut_grad[id001 + shift] += (b_d - r_d) * image_grad[g_index];
                    lut_grad[id101 + shift] += (r_d - g_d) * image_grad[g_index];
                    lut_grad[id111 + shift] += g_d * image_grad[g_index];

                    lut_grad[id000 + shift * 2] += (1 - b_d) * image_grad[b_index];
                    lut_grad[id001 + shift * 2] += (b_d - r_d) * image_grad[b_index];
                    lut_grad[id101 + shift * 2] += (r_d - g_d) * image_grad[b_index];
                    lut_grad[id111 + shift * 2] += g_d * image_grad[b_index];
                }
                else if (r_d <= g_d && b_d > g_d)
                {
                    lut_grad[id000] += (1 - b_d) * image_grad[r_index];
                    lut_grad[id001] += (b_d - g_d) * image_grad[r_index];
                    lut_grad[id011] += (g_d - r_d) * image_grad[r_index];
                    lut_grad[id111] += r_d * image_grad[r_index];

                    lut_grad[id000 + shift] += (1 - b_d) * image_grad[g_index];
                    lut_grad[id001 + shift] += (b_d - g_d) * image_grad[g_index];
                    lut_grad[id011 + shift] += (g_d - r_d) * image_grad[g_index];
                    lut_grad[id111 + shift] += r_d * image_grad[g_index];

                    lut_grad[id000 + shift * 2] += (1 - b_d) * image_grad[b_index];
                    lut_grad[id001 + shift * 2] += (b_d - g_d) * image_grad[b_index];
                    lut_grad[id011 + shift * 2] += (g_d - r_d) * image_grad[b_index];
                    lut_grad[id111 + shift * 2] += r_d * image_grad[b_index];
                }
                else if (r_d <= g_d && b_d > r_d)
                {
                    lut_grad[id000] += (1 - g_d) * image_grad[r_index];
                    lut_grad[id010] += (g_d - b_d) * image_grad[r_index];
                    lut_grad[id011] += (b_d - r_d) * image_grad[r_index];
                    lut_grad[id111] += r_d * image_grad[r_index];

                    lut_grad[id000 + shift] += (1 - g_d) * image_grad[g_index];
                    lut_grad[id010 + shift] += (g_d - b_d) * image_grad[g_index];
                    lut_grad[id011 + shift] += (b_d - r_d) * image_grad[g_index];
                    lut_grad[id111 + shift] += r_d * image_grad[g_index];

                    lut_grad[id000 + shift * 2] += (1 - g_d) * image_grad[b_index];
                    lut_grad[id010 + shift * 2] += (g_d - b_d) * image_grad[b_index];
                    lut_grad[id011 + shift * 2] += (b_d - r_d) * image_grad[b_index];
                    lut_grad[id111 + shift * 2] += r_d * image_grad[b_index];
                }
                else
                {
                    lut_grad[id000] += (1 - g_d) * image_grad[r_index];
                    lut_grad[id010] += (g_d - r_d) * image_grad[r_index];
                    lut_grad[id110] += (r_d - b_d) * image_grad[r_index];
                    lut_grad[id111] += b_d * image_grad[r_index];

                    lut_grad[id000 + shift] += (1 - g_d) * image_grad[g_index];
                    lut_grad[id010 + shift] += (g_d - r_d) * image_grad[g_index];
                    lut_grad[id110 + shift] += (r_d - b_d) * image_grad[g_index];
                    lut_grad[id111 + shift] += b_d * image_grad[g_index];

                    lut_grad[id000 + shift * 2] += (1 - g_d) * image_grad[b_index];
                    lut_grad[id010 + shift * 2] += (g_d - r_d) * image_grad[b_index];
                    lut_grad[id110 + shift * 2] += (r_d - b_d) * image_grad[b_index];
                    lut_grad[id111 + shift * 2] += b_d * image_grad[b_index];
                }
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &tetrahedral_forward, "Tetrahedral forward");
    m.def("backward", &tetrahedral_backward, "Tetrahedral backward");
}

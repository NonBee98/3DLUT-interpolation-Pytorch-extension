#include "trilinear_kernel.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

static const int threads = 1024;

int trilinear_forward_cuda(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                           int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    CHECK_INPUT(lut);
    CHECK_INPUT(image);
    CHECK_INPUT(output);

    const int nElements = height * width * batch;
    cudaError_t err;

    AT_DISPATCH_FLOATING_TYPES(lut.scalar_type(), "trilinear_forward_cuda",
                               ([&]
                                { TriLinearForward<scalar_t>
                                      <<<(nElements + threads - 1) / threads, threads>>>(
                                          nElements,
                                          lut.data_ptr<scalar_t>(),
                                          image.data_ptr<scalar_t>(),
                                          output.data_ptr<scalar_t>(),
                                          lut_dim, shift, binsize, width, height, batch); }));

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return 1;
}

int trilinear_backward_cuda(torch::Tensor image, torch::Tensor image_grad, torch::Tensor lut_grad,
                            int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    CHECK_INPUT(image);
    CHECK_INPUT(image_grad);
    CHECK_INPUT(lut_grad);

    const int nElements = height * width * batch;
    cudaError_t err;

    AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "trilinear_backward_cuda",
                               ([&]
                                { TriLinearBackward<scalar_t>
                                      <<<(nElements + threads - 1) / threads, threads>>>(
                                          nElements,
                                          image.data_ptr<scalar_t>(),
                                          image_grad.data_ptr<scalar_t>(),
                                          lut_grad.data_ptr<scalar_t>(),
                                          lut_dim, shift, binsize, width, height, batch); }));
    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &trilinear_forward_cuda, "Trilinear forward");
    m.def("backward", &trilinear_backward_cuda, "Trilinear backward");
}

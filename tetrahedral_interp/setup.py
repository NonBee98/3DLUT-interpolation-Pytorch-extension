from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

if torch.cuda.is_available():
    print('Including CUDA code.')
    setup(
        name='tetrahedral',
        ext_modules=[
            CUDAExtension('tetrahedral', [
                'src/tetrahedral_cuda.cpp',
                'src/tetrahedral_kernel.cu',
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
else:
    print('NO CUDA is found. Fall back to CPU.')
    setup(name='tetrahedral',
        ext_modules=[CppExtension('tetrahedral', ['src/tetrahedral.cpp'])],
        cmdclass={'build_ext': BuildExtension})

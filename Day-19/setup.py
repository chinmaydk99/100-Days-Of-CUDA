from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="sobel_cuda", #importable python name
    ext_modules=[
        CUDAExtension("sobel_cuda", ["sobel.cpp", "sobel_kernel.cu"]),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)

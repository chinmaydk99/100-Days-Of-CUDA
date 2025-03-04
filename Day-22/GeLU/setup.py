from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="gelu_cuda",
    ext_modules=[
        CUDAExtension("gelu_cuda", ["Gelu.cu"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='flash_fusion',
    ext_modules=[
        CUDAExtension(
            name='flash_fusion',  # This defines TORCH_EXTENSION_NAME
            sources=['mma.cu'],
            extra_compile_args= {
                'nvcc': [
                    "arch=compute_89",
                    "-lineinfo"
                ]
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)

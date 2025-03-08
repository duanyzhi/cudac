# setup.py
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='flash_fusion',
    ext_modules=[
        CppExtension(
            name='flash_fusion',  # This defines TORCH_EXTENSION_NAME
            sources=['mma.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)

# setup.py
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='flow',
    ext_modules=[
        CppExtension(
            name='flow',  # This defines TORCH_EXTENSION_NAME
            sources=['layernorm_40.cu'],
            extra_compile_args={
                'cxx': ['-g'],       # Debug flags for host code
                'nvcc': ['--maxrregcount=32', '-G', '-lineinfo', '--ptxas-options=-v'],  # Debug flags for device code
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)

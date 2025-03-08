# setup.py
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='flow',
    ext_modules=[
        CppExtension(
            name='flow',  # This defines TORCH_EXTENSION_NAME
            sources=['pybind.cc', 'cutlass_layermorm.cu', 'flash_layer_norm.cu', 'layernorm_40.cu'],
            # sources=['layernorm_40.cu', 'lightseq_layernorm.cu'],
            extra_compile_args={
              'cxx': [],
              'nvcc' : [
                '-gencode', 'arch=compute_89,code=sm_89',  # Specify compute capability
                '--ptxas-options=-v',  # Verbose PTX assembly output
                '-O3',  # Optimization level
                '--use_fast_math',  # Use fast math operations
             ],
            }
            # extra_compile_args={
            #     'cxx': ['-g'],       # Debug flags for host code
            #     'nvcc': ['--maxrregcount=32', '-G', '-lineinfo', '--ptxas-options=-v'],  # Debug flags for device code
            # }
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)

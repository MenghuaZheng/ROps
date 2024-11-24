import os
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#mutable-ops

SRC_PATH = []
this_dir = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join(this_dir, "RGemm", "myops", "src")
cpp_sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
cu_sources = list(glob.glob(os.path.join(extensions_dir, "*.cu")))
SRC_PATH.extend(cpp_sources)
SRC_PATH.extend(cu_sources)

print(SRC_PATH)

INCLUDE_DIRS = [
    os.path.join(this_dir, "RGemm", "myops", "include")
]

library_name = "RGemm"

setup(
    name=library_name,
    packages=find_packages(),
    install_requires=["torch"],
    ext_modules=[
        CUDAExtension(
                name=f'{library_name}._C',
                sources=SRC_PATH,
                extra_compile_args={'cxx': ['-g'] + ['-I'+path for path in INCLUDE_DIRS],
                                    'nvcc': ['-O2'] + ['-I'+path for path in INCLUDE_DIRS]},
                extra_link_args=['-Wl,--no-as-needed', '-lcuda'])
    ],
    cmdclass={"build_ext": BuildExtension}
)
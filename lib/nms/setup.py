from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# 確実にディレクトリが存在するようにする
# output_dir = os.path.join(os.path.dirname(__file__), 'nms')
# os.makedirs(output_dir, exist_ok=True)

ext_modules = cythonize([
    Extension(
        ".cpu_nms",
        sources=["cpu_nms.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        ".gpu_nms",
        sources=["gpu_nms.pyx"],
        include_dirs=[numpy.get_include()],
    )
])

setup(
    name="nms",
    ext_modules=ext_modules,
)

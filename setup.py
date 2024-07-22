from setuptools import setup, Extension
import numpy as np
import pybind11

# Define the C++ extension module
matrix_inversion_module = Extension(
    'matrix_inversion',  # Name of the module
    sources=['./matrix_inversion.cpp'],  # Path to your C++ source file
    include_dirs=[np.get_include(), '/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3', pybind11.get_include()],  # Include directories
    extra_compile_args=['-std=c++11'],  # C++ standard
)

setup(
    name='matrix_inversion_package',  # Your package name
    version='0.1',
    description='A package for efficient matrix inversion using Eigen',
    author='Gemma Huai',
    author_email='zhuai@caltech.edu',
    ext_modules=[matrix_inversion_module],  # Register the C++ extension module
    install_requires=[
        'numpy',  # Required dependency
        'pybind11', # pybind11 dependency
    ],
    python_requires='>=3.6',  # Minimum Python version
)

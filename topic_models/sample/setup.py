from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


ext = Extension(
        name='_sample',
        sources=['_sample.pyx'],
        include_dirs=[np.get_include()],
)

setup(
    ext_modules=cythonize(ext),
)

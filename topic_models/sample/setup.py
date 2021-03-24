from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


ext = Extension(
        name='_sample',
        sources=['_sample.pyx'],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    ext_modules=cythonize(ext),
)

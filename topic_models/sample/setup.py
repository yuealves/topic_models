from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


# ext = Extension(
#         name='_sample',
#         sources=['_sample.pyx'],
#         include_dirs=[np.get_include()],
#         define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
# )

ext = [
    Extension(
        name='_sample',
        sources=['_sample.pyx'],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        name='bigdouble',
        sources=['py_big_double.pyx', 'bigdouble.cc'],
        language="c++"
    ),
]

setup(
    ext_modules=cythonize(ext),
)

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


ext = [
    Extension(
        name='topic_models.sample._sample',
        sources=['topic_models/sample/_sample.pyx'],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c++"
    ),
    Extension(
        name='topic_models.utils.bigdouble',
        sources=['topic_models/utils/bigdouble.pyx'],
        language="c++"
    ),
]

setup(
    ext_modules=cythonize(ext),
)


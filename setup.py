from distutils.core import setup, Extension
from glob import glob
from Cython.Build import cythonize
import numpy as np


ext = [
    Extension(
        name="topic_models.sample._sample",
        sources=["topic_models/sample/_sample.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c++"
    ),
    Extension(
        name="topic_models.utils.bigdouble",
        sources=["topic_models/utils/bigdouble.pyx"],
        language="c++"
    ),
]

# include data files specified in MANIFEST.in for building sdist, https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html
# include data files for building a wheel, https://stackoverflow.com/questions/24347450/how-do-you-add-additional-files-to-a-wheel
# https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html
# https://github.com/pypa/sampleproject/blob/main/setup.py
setup(
    ext_modules=cythonize(ext),
    include_package_data=True,
    package_data = {
        "topic_models":["res/*.txt", "res/wiki20/*.txt"]
    }
)


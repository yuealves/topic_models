from libcpp.string cimport string

# Include bigdouble.cc file, details are described in
# https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html
cdef extern from "bigdouble.cc":
    pass

cdef extern from "bigdouble.h":
    cdef cppclass BigDouble:
        BigDouble() except +
        BigDouble(double) except +
        BigDouble(double, int) except +
        void imul(BigDouble *)
        double val
        double get_val()
        int exp
        int get_exp()
        void set_val(double)
        void set_exp(int)
        string repr()


cdef class PyBigDouble:
    cdef BigDouble *thisptr    # hold a C++ instance which we're wrapping

# cdef extern from "bigdouble.cc":
#     pass
from libcpp.string cimport string

cdef extern from "bigdouble.h":
    cdef cppclass BigDouble:
        BigDouble() except +
        BigDouble(double) except +
        BigDouble(double, int) except +
        void imul(BigDouble *)
        double val
        int exp
        void printf()
        string repr()


cdef class PyBigDouble:
    cdef BigDouble *thisptr    # hold a C++ instance which we're wrapping
    def __cinit__(self, double val, int exp):
        self.thisptr = new BigDouble(val, exp)
    def __dealloc__(self):
        del self.thisptr
    def __imul__(self, PyBigDouble b):
        self.thisptr.imul(b.thisptr)
        return self
    def __repr__(self):
        return self.thisptr.repr().decode()
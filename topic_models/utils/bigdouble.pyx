from libcpp.string cimport string
from topic_models.utils.bigdouble cimport BigDouble


cdef class PyBigDouble:
    def __cinit__(self, double val, int exp):
        self.thisptr = new BigDouble(val, exp)
    def __dealloc__(self):
        del self.thisptr
    def __imul__(self, PyBigDouble b):
        self.thisptr.imul(b.thisptr)
        return self
    def __repr__(self):
        return self.thisptr.repr().decode()
    
    # Attribute access
    @property
    def val(self):
        return self.thisptr.get_val()
    @val.setter
    def val(self, val):
        self.thisptr.set_val(val)
    @property
    def exp(self):
        return self.thisptr.get_exp()
    @exp.setter
    def exp(self, exp):
        self.thisptr.set_exp(exp)
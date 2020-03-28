from libcpp.vector cimport vector

cdef extern from "sum_my_vector.h":
    int sum_my_vector(vector[int] my_vector)

def sum_my_vector_cpp(my_list):
    cdef vector[int] my_vector = my_list
    return sum_my_vector(my_vector)
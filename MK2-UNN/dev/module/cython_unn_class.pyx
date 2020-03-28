from libcpp.vector cimport vector

cdef extern from "UNN_link.h":
    cppclass NeuralNetwork:
        int image_channels

		float learning_rate

		int input_nodes
		int output_nodes

		int fcl_idx
		int cnv_idx
		int mxp_idx
		int bnt_idx
		int lrn_inter_idx
		int lrn_intra_idx

		vector[int] fully_connected_nodes_array
		vector[int] conv_nodes_array
		vector[vector[int]] conv_info_array
		vector[int] maxpool_nodes_array
		vector[vector[int]] maxpool_info_array
		vector[vector[float]] LRN_info
		vector[vector[float]] BNT_info

		vector[int] inner_layers

		vector[shared_ptr[Matrix]] weights
		vector[shared_ptr[Matrix]] biases
		vector[vector[float]] bnt_trainables 
		vector[vector[int]] bnt_inner_shapes 

        NeuralNetwork(int input_nodes, int output_nodes, float learning_rate)
        ~NeuralNetwork()

        void handle_trainables(int prev_i, int i)
        void InitializeParameters()

        void FullyConnected(int nodes)
        void Convolutional(const int image_width, const int image_height, const int filter_size, const int filters, const int stride, const int channels)
        void MaxPool(const int filter_size, const int stride)

        void LocalResponseNormalization(const char* type, const float epsilon, const float alpha, const float beta, const float radius)
        void BatchNormalization(const float gamma, const float beta, const double epsilon = .001)        

#should be: cdef class NeuralNetwork
class PythonNetwork:
    cdef NeuralNetwork* obj

    def __init__(self, int input_nodes, int output_nodes, float learning_rate):
        self.cobj = new NeuralNetwork(input_nodes, output_nodes, learning_rate)
        if self.cobj == NULL:
            raise MemoryError('Not enough memory')
    def __del__(self):
        del self.cobj

    def Convolutional(const int image_width, const int image_height, const int filter_size, const int filters, const int stride, const int channels):



def sum_my_vector_cpp(my_list):
    cdef vector[int] my_vector = my_list
    return sum_my_vector(my_vector)

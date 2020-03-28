#include <iostream>
#include <vector>
#include "sum_my_vector.h"
#include "../framework/UNN.h"

NeuralNetwork create_network(int input_nodes, int output_nodes, float learning_rate){
	return std::make_shared<NeuralNetwork>(input_nodes, output_nodes, learning_rate);
}
void convolutional_layer(const int image_width, const int image_height, const int filter_size, const int filters, const int stride, const int channels = 0){
	networks[network_idx].Convolutional(image_width, image_height, filter_size, )
}
void max_pooling_layer(const int filter_size, const int stride){

}
void fully_connected_layer(const int nodes){

}
void local_response_norm(const char* type, const float epsilon, const float alpha, const float beta, const float radius){

}
void batch_norm(const float gamma, const float beta, const double epsilon = .001){

}
void initialize_params(){

	network_idx++;
}
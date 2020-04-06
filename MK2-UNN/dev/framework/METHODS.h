#ifndef METHODS
#define METHODS

#include "STRUCTURE.h"
#include "handle_trainables.h"
#include "predict.h"

void NeuralNetwork::InitializeParameters(){
	handle_trainables(0, inner_layers[0], -1);

	for(int l = 1; l < inner_layers.size() - 1; l++){
		handle_trainables(l, inner_layers[l - 1], inner_layers[l]);
	}

	handle_trainables(inner_layers.size() - 1, -2, inner_layers[inner_layers.size() - 1]);
}

void NeuralNetwork::FullyConnected(const int nodes, const char* activation = "sigmoid"){
	fully_connected_activations.push_back(activation)
	fully_connected_nodes_array.push_back(nodes);
	inner_layers.push_back(0);
}

void NeuralNetwork::Convolutional(const int image_width, const int image_height, const int filter_size, const int filters, const int stride, const int channels = 0) {
	int num_conv_output_vals = filters * pow((int)((in_dim - filter_size)/stride) + 1, 2);
	conv_nodes_array.push_back(num_conv_output_vals);
	std::vector<int> conv_info = {image_width, image_height, filter_size, filters, stride, channels};
	conv_info_array.push_back(conv_info);
	inner_layers.push_back(4);
	if(image_channels == 0 && channels != 0){
		image_channels = channels;
	}
}

void NeuralNetwork::MaxPool(const int filter_size, const int stride) {
	int maxpool_output_vals = image_channels * pow((int)((image[1] - filter_size)/stride) + 1, 2);
	maxpool_nodes_array.push_back(maxpool_output_vals);
	std::vector<int> maxpool_info = {filter_size, stride};
	maxpool_info_array.push_back(maxpool_info);
	inner_layers.push_back(5);
}

void NeuralNetwork::LocalResponseNormalization(const char* type, const int channels, const int width, const int height, const float epsilon, const float alpha, const float beta, const float radius){
	try{
		if(type == "inter_channel"){
			inner_layers.push_back(1);
		}else if(type == "intra_channel"){
			inner_layers.push_back(2);
		}else{
			throw(std::invalid_argument("\nInvalid Local Response Normalization Type. Try any of the following:\n
				\t- inter_channel\n\t- intra_channel\nException thrown in function: NeuralNetwork::LRN()"));
		}

		std::vector<float> inp_vec = {epsilon, alpha, beta, radius}; LRN_info.push_back(inp_vec);
	}
	catch(const std::invalid_argument& e) {
		Logger::Error(e.what())
	}
}

void NeuralNetwork::BatchNormalization(const float gamma, const float beta, const double epsilon = .000000001) {
	inner_layers.push_back(3);
	std::vector<float> inp_vec = {epsilon, gamma, beta}; BNT_info.push_back(inp_vec);
}





#endif


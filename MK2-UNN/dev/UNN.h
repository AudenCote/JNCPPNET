#include "../lib/matrix.h"
#include <string>
#include <vector>
#include <math.h>
#include "LRN.h"
#include "BNT.h"
#include "CNV.h"
#include "FCL.h"

class NeuralNetwork{
private:
	int image_channels = 0;

	int fcl_idx = 0;

	float learning_rate = .1f;

	int input_nodes;
	int output_nodes;

	std::vector<int> fully_connected_nodes_array;
	std::vector<int> conv_nodes_array;
	std::vector<int> conv_inp_array;

	std::vector<int> inner_layers = {};
	//fully connected layer -> 0
	//inter_channel LRN -> 1
	//intra_channel LRN -> 2
	//batch_norm -> 3
	//convolutional layer -> 4
	//max-pooling layer -> 5

	std::vector<std::vector<float>> LRN_inputs;
	std::vector<std::vector<float>> BNT_inputs;

	std::vector<std::shared_ptr<Matrix>> weights;
	std::vector<std::shared_ptr<Matrix>> biases;

	static void check_params(int prev_i, int i){
		try{
			if(i == -2){

			}else if(i == 0){
				if(prev_i == -1){ 
					std::vector<int> w_s = {fully_connected_nodes_array[fcl_idx], input_nodes}; weights.push_back(std::make_shared<Matrix>(w_s));
					std::vector<int> b_s = {, 1}; biases.push_back(std::make_shared<Matrix>(b_s));
				}else if(prev_i == 4){
					std::vector<int> ws = {}; weights.push_back				
				}
				fcl_idx += 1;
			}else if(i == 1){

			}else if(i == 2){

			}else if(i == 3){

			}else if(i == 4){
				if(prev_i == 0){
					throw(std::logic_error("With this network version, convolutional layers may not come after fully connected (dense) layers"));
				}
			}else if(i == 5){
				if(prev_i == 4){

				}else{
					throw(std::logic_errror("With this network version, max-pooling layers must come directly after a convolutional layer, or a normalization layer which comes directly after a convolutional layer"));
				}
			}
		}
		catch(const std::logic_error& e) {
			std::cout << e.what() << "\nException thrown in function: NeuralNetwork::check_params()" << std::endl;
		}
	}
public:
	NeuralNetwork(int in_nodes, int out_nodes, float rate) 
		: input_nodes(in_nodes), output_nodes(out_nodes), learning_rate(rate) { }

	~NeuralNetwork() { }

	void FullyConnected(nodes){

		fully_connected_nodes_array.push_back(nodes);
		inner_layers.push_back(0);

	}

	void Convolutional(const int image_width, const int image_height, const int filter_size, const int filters, const int stride, const int channels = 0) {
		int num_conv_output_vals = filters * pow((int)((in_dim - filter_size)/stride) + 1, 2);
		conv_nodes_array.push_back(num_conv_output_vals);
		conv_inp_array.push_back(conv_inps);
		inner_layers.push_back(4);
		if(image_channels == 0 && channels != 0){
			image_channels = channels;
		}
	}

	void MaxPool(filter_size, stride) {
		int maxpool_output_vals = image_channels * pow((int)((image[1] - filter_size)/stride) + 1, 2);
	}

	void LocalResponseNormalization(const char* type, const float epsilon, const float alpha, const float beta, const float radius){
		try{
			if(type == "inter_channel"){
				inner_layers.push_back(1);
			}else if(type == "intra_channel"){
				inner_layers.push_back(2);
			}else{
				throw std::invalid_argument("\nInvalid Local Response Normalization Type. Try any of the following:\n
					\t- inter_channel\n\t- intra_channel");
			}

			std::vector<float> inp_vec = {epsilon, alpha, beta, radius}; LRN_inputs.push_back(inp_vec);
		}
		catch(const std::invalid_argument& e) {
			std::cout << e.what() << "\nException thrown in function: NeuralNetwork::LRN()" << std::endl;
		}
	}

	void BatchNormalization( const float gamma, const float beta, const double epsilon = .000000001) {
		inner_layers.push_back(3);
		std::vector<float> inp_vec = {epsilon, gamma, beta}; BNT_inputs.push_back(inp_vec);
	}

	void InitializeParameters(){
		check_params(inner_layers[0], -1);

		for(int l = 1; l < inner_layers.size() - 1; l++){
			check_params(inner_layers[l - 1], inner_layers[l]);
		}

		check_params(-2, inner_layers[inner_layers.size() - 1]);
	}

	static std::shared_ptr<Matrix> Predict(const Matrix& input_array, const bool vectorize_inputs = true){
		if(vectorize_inputs){
			std::vector<int> inp_shape = input_array.shape; inp_shape.push_back(1);
			Matrix input_matrix = Matrix(inp_shape); input_matrix.matrix_values = input_array.matrix_values;
		}else{
			Matrix& input_matrix = input_array;
		}


	}
}
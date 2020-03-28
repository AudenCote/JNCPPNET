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

	float learning_rate = .1f;

	int input_nodes;
	int output_nodes;

	//does this idx system work if only updating whenever a new layer of that type is added? 
	int fcl_idx = 0; int cnv_idx = 0; int mxp_idx = 0; int bnt_idx = 0; int lrn_inter_idx = 0; lrn_intra_idx = 0;

	//note - helpful to have all of these separated for debug/dev purposes, maybe make master std::vector later
	std::vector<int> fully_connected_nodes_array;
	std::vector<int> conv_nodes_array; //for use before fully connected layer ONLY 
	std::vector<std::vector<int>> conv_info_array;
	std::vector<int> maxpool_nodes_array; //for use before fully connected layer ONLY 
	std::vector<std::vector<int>> maxpool_info_array;
	std::vector<std::vector<float>> LRN_info;
	std::vector<std::vector<float>> BNT_info;

	std::vector<int> inner_layers = {};
	//fully connected layer -> 0
	//inter_channel LRN -> 1
	//intra_channel LRN -> 2
	//batch_norm -> 3
	//convolutional layer -> 4
	//max-pooling layer -> 5

	std::vector<std::shared_ptr<Matrix>> weights;
	std::vector<std::shared_ptr<Matrix>> biases;
	std::vector<std::vector<float>> bnt_trainables; // gammas and betas for batch norm
	std::vector<std::vector<int>> bnt_inner_shapes; //shape of the layer that comes before batch norm layer 
	// - to be put together with batch size to produce total BNT shape for training

	void handle_trainables(int prev_i, int i){
		try{
			if(i == -2){
				if(prev_i == -1){
					std::vector<int> w_s = {output_nodes, input_nodes}; 
				}else if(prev_i == 0){
					std::vector<int> w_s = {output_nodes, fully_connected_nodes_array[fcl_idx - 1]};  
				}else if(prev_i == 4){
					
				}else if(prev_i == 5){

				}else if(prev_i == 1 || prev_i == 2){

				}else if(prev_i == 3){

				}
				std::vector<int> b_s = {output_nodes, 1}; 
				weights.push_back(std::make_shared<Matrix>(w_s));
				biases.push_back(std::make_shared<Matrix>(b_s));
			}else if(i == 0){
				if(prev_i == -1){
					std::vector<int> w_s = {fully_connected_nodes_array[fcl_idx], input_nodes}; 
				}else if(prev_i == 4){
					std::vector<int> w_s = {fully_connected_nodes_array[fcl_idx], conv_nodes_array[cnv_idx - 1]}; 
				}else if(prev_i == 0){
					std::vector<int> w_s = {fully_connected_nodes_array[fcl_idx], fully_connected_nodes_array[fcl_idx - 1]}
				}else if(prev_i == 1 || prev_i == 2){

				}else if(prev_i == 3){

				}else if(prev_i == 5){

				}
				std::vector<int> b_s = {fully_connected_nodes_array[fcl_idx], 1}; 
				weights.push_back(std::make_shared<Matrix>(w_s));
				biases.push_back(std::make_shared<Matrix>(b_s));
				fcl_idx += 1;
			}else if(i == 1){
				if(prev_i != 4 && prev_i != 5){
					std::cout << "Local response normalization layers are designed primarily to come after a convolutional-type layer" << std::endl;
				}

				if()

				lrn_inter_idx += 1;
			}else if(i == 2){

				lrn_intra_idx += 1;
			}else if(i == 3){
				if(prev_i == -1){
					std::cout << "Batch Normalization is not intended to be used on the input layer - consider randomizing inputs, etc.\nNeural network being structured in accordance with the users preferences" << std::endl;
					std::vector<int> bnt_s = {input_nodes, 1};
				}else if(prev_i == 0){
					std::vector<int> bnt_s = {fully_connected_nodes_array[fcl_idx - 1], 1};
				}else if(prev_i == 1 || prev_i == 2){
					std::cout << "Batch Normalization is not intended to be used in conjunction with local response normalization\nNeural network being structured in accordance with the users preferences" << std::endl;
					std::vector<int> bnt_s = {, 1};
				}
				bnt_inner_shapes.push_back(bnt_s);
				std::vector<float> trainables = {gen_random_float(-1, 1), gen_random_float(-1, 1)}; bnt_trainables.push_back(trainables);
				bnt_idx += 1;
			}else if(i == 4){
				if(prev_i == -1){

				}else if(prev_i == 0){
					throw(std::logic_error("With this network version, convolutional layers may not come after fully connected (dense) layers"));
				}else if(prev_i == 4){
					std::vector<int> w_s = {}; 
					std::vector<int> b_s = {}; 
				}else if(prev_i == 5){

				}else if(prev_i == 1 || prev_i == 2){

				}else if(prev_i == 3){

				}
				std::vector<int> b_s = {conv_info[cnv_idx][3], 1}; //one bias for each filter - same over channels, to put emphasis on features
				weights.push_back(std::make_shared<Matrix>(w_s));
				biases.push_back(std::make_shared<Matrix>(b_s));
				cnv_idx += 1;
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
	NeuralNetwork(const int in_nodes, const int out_nodes, const float rate) 
		: input_nodes(in_nodes), output_nodes(out_nodes), learning_rate(rate) { std::cout << "Neural Network Created" << std::endl; }

	~NeuralNetwork() { std::cout << "Neural Network Destroyed" << std::endl; }

	void InitializeParameters(){
		handle_trainables(inner_layers[0], -1);

		for(int l = 1; l < inner_layers.size() - 1; l++){
			handle_trainables(inner_layers[l - 1], inner_layers[l]);
		}

		handle_trainables(-2, inner_layers[inner_layers.size() - 1]);
	}

	void FullyConnected(int nodes){

		fully_connected_nodes_array.push_back(nodes);
		inner_layers.push_back(0);

	}

	void Convolutional(const int image_width, const int image_height, const int filter_size, const int filters, const int stride, const int channels = 0) {
		int num_conv_output_vals = filters * pow((int)((in_dim - filter_size)/stride) + 1, 2);
		conv_nodes_array.push_back(num_conv_output_vals);
		std::vector<int> conv_info = {image_width, image_height, filter_size, filters, stride, channels};
		conv_info_array.push_back(conv_info);
		inner_layers.push_back(4);
		if(image_channels == 0 && channels != 0){
			image_channels = channels;
		}
	}

	void MaxPool(const int filter_size, const int stride) {
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

			std::vector<float> inp_vec = {epsilon, alpha, beta, radius}; LRN_info.push_back(inp_vec);
		}
		catch(const std::invalid_argument& e) {
			std::cout << e.what() << "\nException thrown in function: NeuralNetwork::LRN()" << std::endl;
		}
	}

	void BatchNormalization( const float gamma, const float beta, const double epsilon = .000000001) {
		inner_layers.push_back(3);
		std::vector<float> inp_vec = {epsilon, gamma, beta}; BNT_info.push_back(inp_vec);
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
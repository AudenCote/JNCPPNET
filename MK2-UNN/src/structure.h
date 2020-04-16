#ifndef STRUCTURE_INCLUDE
#define STRUCTURE_INCLUDE

class NeuralNetwork {
private:
	int image_channels = 0;

	float learning_rate = .1f;

	int input_nodes;
	int output_nodes;

	//does this idx system work if only updating whenever a new layer of that type is added in handle_trainables? 
	int fcl_idx = 0; int cnv_idx = 0; int mxp_idx = 0; int bnt_idx = 0; int lrn_inter_idx = 0; lrn_intra_idx = 0;

	//note - helpful to have all of these separated for debug/dev purposes, maybe make master std::vector later
	std::vector<int> fully_connected_nodes_array;
	std::vector<int> conv_nodes_array; //for use before fully connected layer ONLY 
	std::vector<std::vector<int>> conv_info_array;
	std::vector<int> maxpool_nodes_array; //for use before fully connected layer ONLY 
	std::vector<std::vector<int>> maxpool_info_array;
	std::vector<std::vector<float>> LRN_info;
	std::vector<std::vector<int>> LRN_dimensions;
	std::vector<std::vector<float>> BNT_info;
	std::vector<const char*> fully_connected_activations;

	std::vector<int> inner_layers = {};
	//input layer -> -1
	//output layer -> -2
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

	std::vector<std::shared_ptr<Matrix>> used_weight_values;
	std::vector<std::shared_ptr<Matrix>> used_bias_values;

	void handle_trainables(int prev_i, int i);

	static int get_norm_layer_outputs(const int layer_index);

	static std::vector<std::shared_ptr<Matrix>> feed_forward_all_template(const Matrix& input_array, const bool vectorize_inputs = true);

public:
	NeuralNetwork(const int in_nodes, const int out_nodes, const float rate)
		: input_nodes(in_nodes), output_nodes(out_nodes), learning_rate(rate) {
		Logger::Info("Neural Network Object Created");
	}

	~NeuralNetwork() { Logger::Info("Neural Network Object Destroyed"); }

	void InputLayer(const int in_nodes);

	void OutputLayer(const int out_nodes);

	void InitializeParameters();

	void FullyConnected(int nodes, const char* activation = "sigmoid");

	void Convolutional(const int image_width, const int image_height, const int filter_size, const int filters, const int stride, const int channels = 0, const char* activation = "relu");

	void MaxPool(const int filter_size, const int stride);

	void AvgPool(const int filter_size, const int stride);

	void LocalResponseNormalization(const char* type, const float epsilon, const float alpha, const float beta, const float radius);

	void BatchNormalization(const float gamma, const float beta, const double epsilon = .000000001);

	std::shared_ptr<Matrix> Predict(const Matrix& input_array, const bool vectorize_inputs = true);

	void Train();
}


#endif
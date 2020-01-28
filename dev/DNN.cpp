#include "../lib/matrix.h"


class NeuralNetwork {
private:
	float learning_rate = .1f;

public:
	int input_nodes;
	int output_nodes;
	std::vector<int> hidden_nodes_array;
	int hidden_layers = 0;

	NeuralNetwork(int in_nodes, int out_nodes, float rate) 
		: input_nodes(in_nodes), output_nodes(out_nodes), learning_rate(rate) {

		

	}



}
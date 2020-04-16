#include "../lib/matrix.h"
#include <string>
#include <vector>
#include <math.h>


class NeuralNetwork {
private:
	float learning_rate = .1f;

	int input_nodes;
	int output_nodes;
	std::vector<int> hidden_nodes_array;

	int hidden_layers = 0;

	std::vector<std::shared_ptr<Matrix>> weights;
	std::vector<std::shared_ptr<Matrix>> biases;

	double mean_square_error(Matrix targets, Matrix output) { 
		std::shared_ptr<Matrix> error = Matrix::ElementwiseSubtraction(targets, output);
		error->Square();
		double sum = error->Sum();
		return sum/targets.num_vals;
	}

public:

	NeuralNetwork(int in_nodes, int out_nodes, float rate) 
		: input_nodes(in_nodes), output_nodes(out_nodes), learning_rate(rate) { }

	~NeuralNetwork() { }

	void HiddenLayer(int hidden_nodes) {
		hidden_layers++;
		hidden_nodes_array.push_back(hidden_nodes);
	}

	void InitializeParameters() {

		std::vector<int> w_shape = {hidden_nodes_array[0], input_nodes};
		std::vector<int> b_shape = {hidden_nodes_array[0], 1};
		weights.push_back(std::make_shared<Matrix>(w_shape));
		biases.push_back(std::make_shared<Matrix>(b_shape));

		if(hidden_nodes_array.size() > 1){
			for(int i = 0; i < hidden_layers - 1; ++i) {
				w_shape = {hidden_nodes_array[i + 1], hidden_nodes_array[i]};
				b_shape = {hidden_nodes_array[i + 1], 1};
				weights.push_back(std::make_shared<Matrix>(w_shape));
				biases.push_back(std::make_shared<Matrix>(b_shape));
			}
		}

		w_shape = {output_nodes, hidden_nodes_array[hidden_nodes_array.size() - 1]};
		b_shape = {output_nodes, 1};
		weights.push_back(std::make_shared<Matrix>(w_shape));
		biases.push_back(std::make_shared<Matrix>(b_shape));

		for(std::shared_ptr<Matrix> w_mat : weights)
			w_mat->Randomize();
		for(std::shared_ptr<Matrix> b_mat : biases)
			b_mat->Randomize();

		for(float val : weights[0]->matrix_values){
			std::cout << val << std::endl;
		}
	}

	std::shared_ptr<Matrix> Predict(Matrix& input_array) {
		std::shared_ptr<Matrix> weights_ho = weights[weights.size() - 1];
		std::shared_ptr<Matrix> bias_h = biases[0];
		std::shared_ptr<Matrix> bias_o = biases[biases.size() - 1];

		std::shared_ptr<Matrix> hidden1 = Matrix::DotProduct(*weights[0], input_array);
		hidden1 = Matrix::ElementwiseAddition(*hidden1, *bias_h);
		Matrix::Sigmoid(hidden1);

		std::vector<std::shared_ptr<Matrix>> hiddens= {hidden1};
		for(int i = 0; i < hidden_layers - 1; ++i){
			std::shared_ptr<Matrix> new_hidden = Matrix::DotProduct(*weights[i + 1], *hiddens[hiddens.size() - 1]);
			new_hidden = Matrix::ElementwiseAddition(*new_hidden, *biases[i + 1]);
			Matrix::Sigmoid(new_hidden);
			hiddens.push_back(new_hidden);
		}
		std::shared_ptr<Matrix> outputs = Matrix::DotProduct(*weights_ho, *hiddens[hiddens.size() - 1]);
		outputs = Matrix::ElementwiseAddition(*outputs, *bias_o);
		Matrix::Sigmoid(outputs);

		std::cout << "\nReturning Prediction" << std::endl;
		return outputs;
	}

	void feed_and_propogate(Matrix& input_array, Matrix& target_array, int epochs, int batch_size, bool print = true) {

		if(input_array.shape[0] != target_array.shape[0])
			std::cout << "Different Sample Lengths" << std::endl;

		std::vector<int> input_batches_shape = {(int)ceil(input_array.shape[0]/batch_size), batch_size, input_nodes, 1}; Matrix input_batches(input_batches_shape);
		for(int i = 0; i < input_array.shape[0]; ++i){
			if(i % batch_size == 0){
				for(int j = 0; j < batch_size; ++j){
					for(int k = 0; k < input_nodes; ++k){
						input_batches.Set({(int)ceil(i/batch_size), j, k, 0}, (float)input_array.matrix_values[i + (j*input_nodes) + k]);
					}
				}
			}
		}
		
		std::vector<int> target_batches_shape = {(int)ceil(target_array.shape[0]/batch_size), batch_size, output_nodes, 1}; Matrix target_batches(target_batches_shape);
		for(int i = 0; i < target_array.shape[0]; ++i){
			if(i % batch_size == 0)
				for(int j = 0; j < batch_size; ++j){
					for(int k = 0; k < output_nodes; ++k){
						target_batches.Set({(int)ceil(i/batch_size), j, k, 0}, (float)target_array.matrix_values[i + (j*output_nodes) + k]);
					}
				}
		}

		std::vector<double> losses;
		for(int e = 0; e < epochs; ++e){

			for(int b = 0; b < input_batches.shape[0]; ++b){

				std::shared_ptr<Matrix> input_batch = input_batches.GetChunk({b});
				std::shared_ptr<Matrix> target_batch = target_batches.GetChunk({b});

				std::vector<std::vector<std::shared_ptr<Matrix>>> bias_deltas;
				std::vector<std::vector<std::shared_ptr<Matrix>>> weights_deltas;

				double loss;

				for(int p = 0; p < batch_size; ++p){
					std::vector<std::shared_ptr<Matrix>> sample_weights_deltas; std::vector<std::shared_ptr<Matrix>> sample_bias_deltas;

					std::shared_ptr<Matrix> inputs = input_batch->GetChunk({p});
					std::shared_ptr<Matrix> targets = target_batch->GetChunk({p});

					std::shared_ptr<Matrix> hidden1 = Matrix::DotProduct(*weights[0], *inputs);

					hidden1 = Matrix::ElementwiseAddition(*hidden1, *biases[0]);
					Matrix::Sigmoid(hidden1);

					std::vector<std::shared_ptr<Matrix>> hiddens = {hidden1};
					for(int i = 0; i < hidden_layers - 1; ++i){
						std::shared_ptr<Matrix> new_hidden = Matrix::DotProduct(*weights[i + 1], *hiddens[hiddens.size() - 1]);
						new_hidden = Matrix::ElementwiseAddition(*new_hidden, *biases[i + 1]);
						Matrix::Sigmoid(new_hidden);
						hiddens.push_back(new_hidden);
					}

					std::shared_ptr<Matrix> outputs = Matrix::DotProduct(*weights[weights.size() - 1], *hiddens[hiddens.size() - 1]);
					outputs = Matrix::ElementwiseAddition(*outputs, *biases[biases.size() - 1]);
					Matrix::Sigmoid(outputs);

					loss = NeuralNetwork::mean_square_error(*targets, *outputs);
					std::shared_ptr<Matrix> last_errors = Matrix::ElementwiseSubtraction(*targets, *outputs);
					Matrix::SigmoidPrime(outputs);
					std::shared_ptr<Matrix> gradients = Matrix::ElementwiseMultiplication(*outputs, *last_errors);
					gradients->Multiply(learning_rate);

					std::shared_ptr<Matrix> hidden3_tr = Matrix::Transpose(hiddens[hiddens.size() - 1]);
					std::shared_ptr<Matrix> weights_ho_deltas = Matrix::DotProduct(*gradients, *hidden3_tr);

					sample_weights_deltas.push_back(weights_ho_deltas);
					sample_bias_deltas.push_back(gradients);
					
					for(int i = 0; i < hidden_layers - 1; ++i){
						std::shared_ptr<Matrix> current = weights[weights.size() -(i+1)];
						std::shared_ptr<Matrix> new_hidden = hiddens[hiddens.size()-(i+2)];

						std::shared_ptr<Matrix> current_transposed = Matrix::Transpose(current);
						last_errors = Matrix::DotProduct(*current_transposed, *last_errors);
						Matrix::SigmoidPrime(hiddens[hiddens.size()-(i+1)]);
						std::shared_ptr<Matrix> gradients = Matrix::ElementwiseMultiplication(*hiddens[hiddens.size()-(i+1)], *last_errors);
						gradients->Multiply(learning_rate);
						

						std::shared_ptr<Matrix> new_hidden_transposed = Matrix::Transpose(new_hidden);
						std::shared_ptr<Matrix> deltas = Matrix::DotProduct(*gradients, *new_hidden_transposed);

						sample_weights_deltas.push_back(deltas);
						sample_bias_deltas.push_back(gradients);
					}

					std::shared_ptr<Matrix> weights1_tr = Matrix::Transpose(weights[1]);
					//THIS RETURNS ZEROES
					std::shared_ptr<Matrix> hidden1_errors = Matrix::DotProduct(*weights1_tr, *last_errors); 

					Matrix::SigmoidPrime(hidden1);
					std::shared_ptr<Matrix> hidden1_gradient = Matrix::ElementwiseMultiplication(*hidden1, *hidden1_errors);
					hidden1_gradient->Multiply(learning_rate);

					std::shared_ptr<Matrix> inputs_tr = Matrix::Transpose(inputs);
					std::shared_ptr<Matrix> weight_ih_deltas = Matrix::DotProduct(*hidden1_gradient, *inputs_tr);

					sample_weights_deltas.push_back(weight_ih_deltas);
					sample_bias_deltas.push_back(hidden1_gradient);

					weights_deltas.push_back(sample_weights_deltas);
					bias_deltas.push_back(sample_bias_deltas);
				}

				std::vector<std::shared_ptr<Matrix>> summed_weights_deltas = {};
				for(std::shared_ptr<Matrix> w : weights){
					summed_weights_deltas.push_back(std::make_shared<Matrix>(w->shape));
				}
				std::vector<std::shared_ptr<Matrix>> summed_bias_deltas = {};
				for(std::shared_ptr<Matrix> b : biases){
				 	summed_bias_deltas.push_back(std::make_shared<Matrix>(b->shape));
				}

				for(int i = 0; i < summed_weights_deltas.size(); ++i){
					summed_weights_deltas[i]->Zero(); summed_bias_deltas[i]->Zero();
				}

				for(int s = 0; s < batch_size; ++s) {
					for(int l = 0; l < summed_weights_deltas.size(); ++l){
						summed_weights_deltas[l] = Matrix::ElementwiseAddition(*summed_weights_deltas[l], *weights_deltas[s][weights_deltas[s].size() - 1 - l]);
					}
				}
				for(int s = 0; s < batch_size; ++s) {
					for(int l = 0; l < summed_bias_deltas.size(); ++l){
						summed_bias_deltas[l] = Matrix::ElementwiseAddition(*summed_bias_deltas[l], *bias_deltas[s][bias_deltas[s].size() - 1 - l]);
					}
				}

				for(int l = 0; l < summed_weights_deltas.size(); ++l){
					summed_weights_deltas[l]->Divide(batch_size);
					summed_bias_deltas[l]->Divide(batch_size);
				}

				weights[weights.size() - 1] = Matrix::ElementwiseAddition(*weights[weights.size() - 1], *summed_weights_deltas[summed_weights_deltas.size()-1]);
				biases[biases.size() - 1] = Matrix::ElementwiseAddition(*biases[biases.size() - 1], *summed_bias_deltas[summed_bias_deltas.size()-1]);
				for(int i = 0; i < hidden_layers - 1; ++i){
					weights[weights.size()-(i + 2)] = Matrix::ElementwiseAddition(*weights[weights.size()-(i + 2)], *summed_weights_deltas[summed_weights_deltas.size()-(i+2)]);
					biases[biases.size()-(i + 2)] = Matrix::ElementwiseAddition(*biases[biases.size()-(i + 2)], *summed_bias_deltas[summed_bias_deltas.size()-(i+2)]);
				}
				weights[0] = Matrix::ElementwiseAddition(*weights[0], *summed_weights_deltas[0]);
				biases[0] = Matrix::ElementwiseAddition(*biases[0], *summed_bias_deltas[0]);

				if(print){
					std::cout << "\n" << "================" << std::endl;
					std::cout << "Epoch: " << e + 1 << std::endl;
					std::cout << "Number of Epochs: " << epochs << std::endl;
					std::cout << "Batch: " << b + 1 << std::endl;
					std::cout << "Number of Batches: " << ceil(input_array.shape[0]/batch_size) << std::endl;
					std::cout << "Batch Size: " << batch_size << std::endl;
					std::cout << "Loss: " << loss << std::endl;
					std::cout << "================" << "\n" << std::endl;
				}
			}
		}
	}
};
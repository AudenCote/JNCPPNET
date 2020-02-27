#include "../lib/matrix.h"
#include <string>
#include <vector>
#include <math.h>


class NeuralNetwork {
private:
	float learning_rate = .1f;

	float mean_square_error(Matrix targets, Matrix output) { 
		Matrix *error = Matrix::ElementwiseSubtraction(targets, output);
		error->Square();
		float sum = error->Sum();
		return sum/targets.num_vals;
	}

	void deallocate_wb(){
		std::cout << "In function" << std::endl;

		for(Matrix* w : weights){ delete w; }
		for(Matrix* b : biases){ delete b; }
	}

public:
	int input_nodes;
	int output_nodes;
	std::vector<int> hidden_nodes_array;
	int hidden_layers = 0;
	std::vector<Matrix*> weights;
	std::vector<Matrix*> biases;

	NeuralNetwork(int in_nodes, int out_nodes, float rate) 
		: input_nodes(in_nodes), output_nodes(out_nodes), learning_rate(rate) { }

	~NeuralNetwork() {
		std::cout << "Deallocating" << std::endl;
		deallocate_wb();
	}

	void HiddenLayer(int hidden_nodes) {
		hidden_layers++;
		hidden_nodes_array.push_back(hidden_nodes);
	}

	void InitializeParameters() {

		std::vector<int> w_shape = {hidden_nodes_array[0], input_nodes};
		std::vector<int> b_shape = {hidden_nodes_array[0], 1};
		weights.push_back(new Matrix(w_shape));
		biases.push_back(new Matrix(b_shape));

		if(hidden_nodes_array.size() > 1){
			std::cout << "Yes" << std::endl;
			for(int i = 0; i < hidden_layers; ++i) {
				w_shape = {hidden_nodes_array[i + 1], hidden_nodes_array[i]};
				b_shape = {hidden_nodes_array[i + 1], 1};
				weights.push_back(new Matrix(w_shape));
				biases.push_back(new Matrix(b_shape));
			}
		}

		w_shape = {output_nodes, hidden_nodes_array[hidden_nodes_array.size() - 1]};
		b_shape = {output_nodes, 1};
		weights.push_back(new Matrix(w_shape));
		biases.push_back(new Matrix(b_shape));

		for(Matrix* w_mat : weights)
			w_mat->Randomize();
		for(Matrix* b_mat : biases)
			b_mat->Randomize();
	}

	Matrix *Predict(Matrix& input_array) {

		Matrix* weights_ih = weights[0];
		Matrix* weights_ho = weights[weights.size() - 1];
		Matrix* bias_h = biases[0];
		Matrix* bias_o = biases[biases.size() - 1];

		Matrix* hidden1 = Matrix::DotProduct(*weights_ih, input_array);
		hidden1 = Matrix::ElementwiseAddition(*hidden1, *bias_h);
		Matrix::Sigmoid(hidden1);
		Matrix*& new_hidden = hidden1;

		std::vector<Matrix*> hiddens= {new_hidden};
		for(int i = 0; i < hidden_layers - 1; ++i){
			new_hidden = Matrix::DotProduct(*weights[i + 1], *hiddens[hiddens.size() - 1]);
			new_hidden = Matrix::ElementwiseAddition(*new_hidden, *biases[i + 1]);
			Matrix::Sigmoid(new_hidden);
			hiddens.push_back(new_hidden);
		}
		Matrix *outputs = Matrix::DotProduct(*weights_ho, *hiddens[hiddens.size() - 1]);
		outputs = Matrix::ElementwiseAddition(*outputs, *bias_o);
		Matrix::Sigmoid(outputs);

		for(Matrix* h : hiddens){ delete h; }
		return outputs;
	}

	void feed_and_propogate(Matrix& input_array, Matrix& target_array, int epochs, int batch_size) {

		//ALMOST ALL ISSUES IN BELOW CODE CHUNK

		std::vector<int> input_shape = {input_array.shape[0], 2, 1}; Matrix input_all(input_shape);
		for(int i = 0; i < input_all.shape[0]; ++i){
			if(i % 2 == 0) input_all.memPtr[i] = input_array.memPtr[i]; input_all.memPtr[i + 1] = target_array.memPtr[i];
		}


		std::vector<int> batches_shape = {(int)ceil(input_all.shape[0]/batch_size), batch_size, 2, 1}; Matrix batches(batches_shape);
		for(int i = 0; i < input_all.shape[0]; ++i){
			if(i % batch_size == 0)
				for(int j = i; j < batch_size + i; ++j){
					for(int k = 0; k < 2; ++k){
						std::cout << "CP-1" << std::endl;
						batches.Set({(int)ceil(i/batch_size), j, k, 0}, input_all.memPtr[i + j + k]);
					}
				}
		}



		std::cout << "CP0" << std::endl; 

		std::vector<float> losses;
		for(int e = 0; e < epochs; ++e){

			for(int b = 0; b < batches.num_vals; ++b){
				Matrix* batch = batches.GetChunk({b});

				std::vector<std::vector<Matrix*>> bias_deltas;
				std::vector<std::vector<Matrix*>> weights_deltas;

				for(int p = 0; p < batch->num_vals; ++p){

					std::vector<Matrix*> sample_weights_deltas; std::vector<Matrix*> sample_bias_deltas;

					Matrix* it_pair = batch->GetChunk({p});
					Matrix* inputs = it_pair->GetChunk({0}); Matrix* targets = it_pair->GetChunk({1});

					Matrix* hidden1 = Matrix::DotProduct(*weights[0], *inputs);
					hidden1 = Matrix::ElementwiseAddition(*hidden1, *biases[0]);
					Matrix::Sigmoid(hidden1);
					Matrix*& new_hidden = hidden1;

					std::vector<Matrix*> hiddens = {new_hidden};
					for(int i = 0; i < hidden_layers - 1; ++i){
						Matrix *new_hidden = Matrix::DotProduct(*weights[i + 1], *hiddens[hiddens.size() - 1]);
						new_hidden = Matrix::ElementwiseAddition(*new_hidden, *biases[i + 1]);
						Matrix::Sigmoid(new_hidden);
						hiddens.push_back(new_hidden);
					}
					Matrix *outputs = Matrix::DotProduct(*weights[weights.size() - 1], *hiddens[hiddens.size() - 1]);
					outputs = Matrix::ElementwiseAddition(*outputs, *biases[biases.size() - 1]);
					Matrix::Sigmoid(outputs);

					//GRADIENT DESCENT STARTS HERE

					float loss = NeuralNetwork::mean_square_error(*targets, *outputs);
					Matrix* last_errors = Matrix::ElementwiseSubtraction(*targets, *outputs);
					Matrix::SigmoidPrime(outputs);
					Matrix* gradients = Matrix::ElementwiseMultiplication(*outputs, *last_errors);
					gradients->Multiply(learning_rate);

					Matrix* hidden3_tr = Matrix::Transpose(hiddens[hiddens.size() - 1]);
					Matrix* weights_ho_deltas = Matrix::DotProduct(*gradients, *hidden3_tr);

					sample_weights_deltas.push_back(weights_ho_deltas);
					sample_bias_deltas.push_back(gradients);
					delete gradients; delete hidden3_tr; delete weights_ho_deltas;

					for(int i = 0; i < hidden_layers; ++i){
						Matrix* current = weights[-(i+1)];
						Matrix& new_hidden = *hiddens[-(i+2)];

						Matrix* current_transposed = Matrix::Transpose(current);
						last_errors = Matrix::DotProduct(*current_transposed, *last_errors);
						Matrix::SigmoidPrime(hiddens[-(i+1)]);
						Matrix* gradients = Matrix::ElementwiseMultiplication(*hiddens[-(i+1)], *last_errors);
						gradients->Multiply(learning_rate);

						Matrix* new_hidden_transposed = Matrix::Transpose(&new_hidden);
						Matrix* deltas = Matrix::DotProduct(*gradients, *new_hidden_transposed);

						sample_weights_deltas.push_back(deltas);
						sample_bias_deltas.push_back(gradients);
						delete current_transposed; delete gradients; delete new_hidden_transposed; delete deltas;
					}

					Matrix* weights1_tr = Matrix::Transpose(weights[1]);
					Matrix* hidden1_errors = Matrix::DotProduct(*weights1_tr, *last_errors); 

					Matrix::SigmoidPrime(hidden1);
					Matrix* hidden1_gradient = Matrix::ElementwiseMultiplication(*hidden1, *hidden1_errors);
					hidden1_gradient->Multiply(learning_rate);

					Matrix* inputs_tr = Matrix::Transpose(inputs);
					Matrix* weight_ih_deltas = Matrix::DotProduct(*hidden1_gradient, *inputs_tr);

					sample_weights_deltas.push_back(weight_ih_deltas);
					sample_bias_deltas.push_back(hidden1_gradient);
					delete weights1_tr; delete hidden1_errors; delete last_errors; delete hidden1_gradient; delete inputs_tr; delete weight_ih_deltas;

					weights_deltas.push_back(sample_weights_deltas);
					bias_deltas.push_back(sample_bias_deltas);
					for(Matrix* h : hiddens) { delete h; }
					delete inputs; delete targets; delete outputs;
				}
				Matrix* summed_weights_deltas = new Matrix(batch->shape);
				Matrix* summed_bias_deltas = new Matrix(batch->shape);
				for(int s = 0; s < weights_deltas.size(); ++s) {
					for(int i = 0; i < summed_weights_deltas->shape[0]; ++i){
						summed_weights_deltas = Matrix::ElementwiseAddition(*summed_weights_deltas->GetChunk({i}), *weights_deltas[s][i]);
					}
				}
				for(int s = 0; s < bias_deltas.size(); ++s) {
					for(int i = 0; i < summed_weights_deltas->shape[0]; ++i){
						summed_bias_deltas = Matrix::ElementwiseAddition(*summed_bias_deltas->GetChunk({i}), *bias_deltas[s][i]);
					}
				}
				summed_weights_deltas->Divide(batch_size);
				summed_bias_deltas->Divide(batch_size);

				weights[weights.size() - 1] = Matrix::ElementwiseAddition(*weights[weights.size() - 1], *summed_weights_deltas->GetChunk({0}));
				biases[biases.size() - 1] = Matrix::ElementwiseAddition(*biases[biases.size() - 1], *summed_bias_deltas->GetChunk({0}));

				for(int i = 0; i < hidden_layers; ++i){
					weights[-(i + 2)] = Matrix::ElementwiseAddition(*weights[-(i + 2)], *summed_weights_deltas->GetChunk({i + 1}));
					biases[-(i + 1)] = Matrix::ElementwiseAddition(*biases[-(i + 2)], *summed_bias_deltas->GetChunk({i + 1}));
				}
				weights[0] = Matrix::ElementwiseAddition(*weights[0], *summed_weights_deltas->GetChunk({summed_weights_deltas->shape[0] - 1}));
				biases[0] = Matrix::ElementwiseAddition(*biases[0], *summed_bias_deltas->GetChunk({summed_weights_deltas->shape[0] - 1}));

				delete summed_weights_deltas;
				delete summed_bias_deltas;	
			}
		}
	}
};


















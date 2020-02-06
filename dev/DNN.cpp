#include "../lib/matrix.h"
#include <string>


class NeuralNetwork {
private:
	float learning_rate = .1f;

	float mean_square_error(Matrix targets, Matrix output) { 
		Matrix *error = Matrix::ElementwiseSubtraction(targets, output);
		error->Square();
		float sum = error->Sum();
		return sum/targets.num_vals;
	}

public:
	int input_nodes;
	int output_nodes;
	std::vector<int> hidden_nodes_array;
	int hidden_layers = 0;
	std::vector<Matrix> weights;
	std::vector<Matrix> biases;

	NeuralNetwork(int in_nodes, int out_nodes, float rate) 
		: input_nodes(in_nodes), output_nodes(out_nodes), learning_rate(rate) { }

	void HiddenLayer(int hidden_nodes) {
		hidden_layers++;
		hidden_nodes_array.push_back(hidden_nodes);
	}

	void InitializeParameters() {
		std::vector<int> w_shape = {hidden_nodes_array[0], input_nodes};
		std::vector<int> b_shape = {hidden_nodes_array[0], 1};
		weights.push_back(Matrix(w_shape));
		weights.push_back(Matrix(b_shape));

		for(int i = 0; i < hidden_layers; ++i) {
			w_shape = {hidden_nodes_array[i + 1], hidden_nodes_array[i]};
			b_shape = {hidden_nodes_array[i + 1], 1};
			weights.push_back(w_shape);
			biases.push_back(b_shape);
		}

		w_shape = {output_nodes, hidden_nodes_array[hidden_nodes_array.size() - 1]};
		b_shape = {output_nodes, 1};
		weights.push_back(w_shape);
		biases.push_back(b_shape);

		for(Matrix w_mat : weights)
			w_mat.Randomize();
		for(Matrix b_mat : biases)
			b_mat.Randomize();
	}

	Matrix *Predict(Matrix& input_array) {

		Matrix& weights_ih = weights[0];
		Matrix& weights_ho = weights[weights.size() - 1];
		Matrux& bias_h = biases[0];
		Matrix& bias_o = biases[biases.size() - 1];

		Matrix* new_hidden = Matrix::DotProduct(weights_ih, input_array);
		new_hidden = Matrix::ElementwiseAddition(*hidden1, bias_h);
		Matrix::Sigmoid(new_hidden);

		std::vector<Matrix*> hiddens= {new_hidden};
		for(int i = 0; i < hidden_layers - 1; ++i)
			new_hidden = Matrix::DotProduct(weights[i + 1], hiddens[hiddens.size() - 1]);
			new_hidden = Matrix::ElementwiseAddition(*new_hidden, biases[i + 1]);
			Matrix::Sigmoid(new_hidden);
			hiddens.push_back(new_hidden);

		Matrix *outputs = Matrix::DotProduct(weights_ho, hiddens[hiddens.size() - 1]);
		outputs = Matrix::ElementwiseAddition(*outputs, bias_o);
		Matrix::Sigmoid(outputs);

		return outputs;
	}

	void feed_and_propogate(Matrix& input_array, Matrix& target_array, int epochs, int batch_size) {
		std::vector<int> input_shape = {input_array.shape[0], 2, 1}; Matrix input_all(input_shape);
		for(int i = 0; i < input_all.shape[0]; ++i)
			if(i % 2 == 0) input_all.memPtr[i] = input_array.memPtr[i]; input_all.memPtr[i + 1] = target_array.memPtr[i]

		std::vector<int> batches_shape = {ciel(input_all.shape[0]/batch_size), batch_size, 2, 1}; Matrix batches(batches_shape);
		for(int i = 0; i < input_all.shape[0]; ++i){
			if(i % batch_size == 0)
				for(int j = i; j < batch_size + i; ++j){
					for(int k = 0; k < 2; ++k)
						batches.Set({ciel(i/batch_size), j, k, 0}, input_all.memPtr[i + j + k]);
				}
		}

		std::vector<float> losses;
		for(int e = 0; e < epochs; ++e){

			for(int b = 0; b < batches.num_vals; ++b){

				Matrix* batch = batches.GetChunk[b];

				for(int p = 0; p < batch.num_vals; ++p){

					std::vector<Matrix*> sample_weights_deltas; std::vector<Matrix*> sample_bias_deltas;

					Matrix* it_pair = batch.GetChunk[p]
					Matrix* inputs = it_pair.GetChunk[0]; Matrix* targets = it_pair.GetChunk[1];

					Matrix* new_hidden = Matrix::DotProduct(weights[0], inputs);
					new_hidden = Matrix::ElementwiseAddition(hidden1, biases[0]);
					Matrix::Sigmoid(new_hidden);

					std::vector<Matrix*> hiddens= {new_hidden};
					for(int i = 0; i < hidden_layers - 1; ++i)
						Matrix *new_hidden = Matrix::DotProduct(weights[i + 1], hiddens[hiddens.size() - 1]);
						new_hidden = Matrix::ElementwiseAddition(*new_hidden, biases[i + 1]);
						Matrix::Sigmoid(new_hidden);
						hiddens.push_back(new_hidden);

					Matrix *outputs = Matrix::DotProduct(weights_ho, hiddens[hiddens.size() - 1]);
					outputs = Matrix::ElementwiseAddition(*outputs, bias_o);
					Matrix::Sigmoid(outputs);

					//GRADIENT DESCENT STARTS HERE

					float loss = NeuralNetwork::mean_square_error(targets, outputs);
					Matrix* last_errors = Matrix::ElementwiseSubtraction(targets, outputs);
					Matrix::SigmoidPrime(outputs);
					Matrix* gradients = ElementwiseMultiplication(outputs, last_errors);
					gradients.Multiply(learning_rate);

					Matrix* hidden3_t = Matrix::Transpose(hiddens[hiddens.size() - 1]);
					weights_ho_deltas = Matrix::DotProduct(*gradient, *hidden3_t);

					sample_weights_deltas.push_back(weights_ho_deltas);
					sample_bias_deltas.push_back(gradients);

					for(int i = 0; i < hidden_layers; ++i){

						Matrix current& = weights[-(i+1)];
						Matrix new_hidden = *hiddens[-(i+2)];

						Matrix* current_transposed = Matrix::Transpose(&current);
						Matrix* last_errors = Matrix::DotProduct(current_transposed, last_errors);
						Matrix::SigmoidPrime(hiddens[-(i+1)]);
						Matrix* gradients = ElementwiseMultiplication(hiddens[-(i+1)], last_errors);
						gradients.Multiply(learning_rate);

						



					}


				}
				
			}

		}

	

	}

};


int main() {
	return 0;
}


















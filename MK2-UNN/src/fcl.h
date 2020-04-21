#ifndef FCL_INCLUDE
#define FCL_INCLUDE

namespace fully_connected {

	//one hidden/output fully connected layer - takes in input values, weights for that layer, 
	//biases for that layer, and non-linearity type, and returns output values for that layer
	std::shared_ptr<Matrix> feed_forward(Matrix& inputs, Matrix& weights, Matrix& biases, const char* activation) {
		try {
			std::shared_ptr<Matrix> output = Matrix::ElementwiseAddition(*Matrix::DotProduct(weights, inputs), biases);

			if (activation == "sigmoid" || activation == "Sigmoid") {
				Matrix::Sigmoid(output);
			}
			else if (activation == "ReLU" || activation == "relu" || activation == "Relu") {
				Matrix::ReLU(output);
			}
			else if (activation == "softmax" || activation == "Softmax") {
				Matrix::Softmax(output);
			}
			else {
				throw(std::invalid_argument("Invalid non-linear (activation) function type\nException thrown in function: fully_connected::feed_forward()"));
			}

			return output;
		}
		catch (const std::invalid_argument& e) {
			Logger::Error(e.what());
			return nullptr;
		}
	}

	std::vector<std::shared_ptr<Matrix>> backprop(std::shared_ptr<Matrix>& last_errors, std::shared_ptr<Matrix> weight_m, std::shared_ptr<Matrix> curr_hidden, std::shared_ptr<Matrix> new_hidden, float learning_rate, const char* activation) {
		std::shared_ptr<Matrix> current_transposed = Matrix::Transpose(weight_m);
		last_errors = Matrix::DotProduct(*current_transposed, *last_errors);
		Matrix::SigmoidPrime(curr_hidden);
		std::shared_ptr<Matrix> gradients = Matrix::ElementwiseMultiplication(*curr_hidden, *last_errors);
		gradients->Multiply(learning_rate);

		std::shared_ptr<Matrix> new_hidden_transposed = Matrix::Transpose(new_hidden);
		std::shared_ptr<Matrix> deltas = Matrix::DotProduct(*gradients, *new_hidden_transposed);

		std::vector<std::shared_ptr<Matrix>> return_vec = { deltas, gradients };  return return_vec;
	}

}


#endif
#ifndef FCL_INCLUDE
#define FCL_INCLUDE

namespace fully_connected {

	//one hidden/output fully connected layer - takes in input values, weights for that layer, 
	//biases for that layer, and non-linearity type, and returns output values for that layer
	std::shared_ptr<Matrix> feed_forward(const Matrix& inputs, const Matrix& weights, const Matrix& biases, const char* activation = "sigmoid") {
		try {
			std::shared_ptr<Matrix> output = Matrix::ElementwiseAddition(Matrix::DotProduct(weights, inputs), biases);
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
		}
		catch (const std::invalid_argument& e) {
			Logger::Error(e.what());
			return nullptr;
		}

		return output;
	}

	std::shared_ptr<Matrix> backprop(Matrix& training_data, Matrix& target_data) {



	}

	}


#endif
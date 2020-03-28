#include "../lib/matrix.h"

class fully_connected{

	fully_connected() { std::cout << "Improper use of fully_connected class" << std::endl; }

	~fully_connected() { }

	//one hidden/output fully connected layer - takes in input values, weights for that layer, 
	//biases for that layer, and non-linearity type, and returns output values for that layer
	static std::shared_ptr<Matrix> feed_forward (const Matrix& inputs, const Matrix& weights, const Matrix& biases, const char* activation = "sigmoid") {
		try{
			std::shared_ptr<Matrix> output = Matrix::DotProduct(weights, inputs);
			output = Matrix::ElementwiseAddition(output, biases);
			if(activation == "sigmoid"){
				Matrix::Sigmoid(output);
			}else if(activation == "ReLU"){

			}else{
				throw(std::invalid_argument("Invalid non-linear (activation) function type\nException thrown in function: fully_connected::feed_forward()"));
			}
		}
		catch (const std::invalid_argument& e) {
			Logger::Error(e.what());
			return nullptr; 
		}

		return output;
	}

}
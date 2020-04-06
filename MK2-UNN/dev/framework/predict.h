#ifndef PREDICT_INCLUDE
#define PREDICT_INCLUDE


static std::shared_ptr<Matrix> Predict(const Matrix& input_array, const bool vectorize_inputs = true){
	if(vectorize_inputs){
		std::vector<int> inp_shape = input_array.shape; inp_shape.push_back(1);
		Matrix input_matrix = Matrix(inp_shape); input_matrix.matrix_values = input_array.matrix_values;
	}else{
		Matrix& input_matrix = input_array;
	}


	if(inner_layers[0] == 0){
		std::shared_ptr<Matrix> hidden1 = fully_connected::feed_forward(input_matrix, weights[0], biases[0], fully_connected_activations[0]);
	}else if(inner_layers[0] == 1){
		std::shared_ptr()
	}




}

#endif
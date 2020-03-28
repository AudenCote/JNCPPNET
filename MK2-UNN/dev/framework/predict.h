#ifndef PREDICT_INCLUDE
#define PREDICT_INCLUDE


static std::shared_ptr<Matrix> Predict(const Matrix& input_array, const bool vectorize_inputs = true){
	if(vectorize_inputs){
		std::vector<int> inp_shape = input_array.shape; inp_shape.push_back(1);
		Matrix input_matrix = Matrix(inp_shape); input_matrix.matrix_values = input_array.matrix_values;
	}else{
		Matrix& input_matrix = input_array;
	}


}

#endif
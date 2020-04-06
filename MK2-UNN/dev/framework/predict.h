#ifndef PREDICT_INCLUDE
#define PREDICT_INCLUDE


static std::shared_ptr<Matrix> Predict(const Matrix& input_array, const bool vectorize_inputs = true){
	if(vectorize_inputs){
		std::vector<int> inp_shape = input_array.shape; inp_shape.push_back(1);
		Matrix input_matrix = Matrix(inp_shape); input_matrix.matrix_values = input_array.matrix_values;
	}else{
		Matrix& input_matrix = input_array;
	}

	lrn_predict_idx = 0; // keeping track of how many times the next layer has been an lrn layer so that the dims (channels, width, height) 
	//and the info (epsilon, radius, etc. ) can be accessed
	fcl_idx = 0; //same idea with other layers



	if(inner_layers[0] == 0){
		std::shared_ptr<Matrix> last_hidden = fully_connected::feed_forward(input_matrix, weights[0], biases[0], fully_connected_activations[0]);
	}else if(inner_layers[0] == 1){
		std::shared_ptr<Matrix> last_hidden = LRN::inter_channel(input_matrix, LRN_dimensions[lrn_predict_idx][0], LRN_dimensions[lrn_predict_idx][1], 
															 LRN_dimensions[lrn_predict_idx][2], LRN_info[lrn_predict_idx][0], LRN_info[lrn_predict_idx][1], 
															 LRN_info[lrn_predict_idx][2], LRN_info[lrn_predict_idx][3]);
		lrn_idx += 1;
	}else if(inner_layers[0] == 2){

	}else if(inner_layers[0] == 3){

	}else if(inner_layers[0] == 4){

	}else if(inner_layers[0] == 5){

	}


	for(int l = 0; l < inner_layers.size(); l++){
		if(inner_layers[0] == 0){
			std::shared_ptr<Matrix> last_hidden = fully_connected::feed_forward(last_hidden, weights[l + 1], biases[l + 1], fully_connected_activations[]);
		}else if(inner_layers[0] == 1){
			std::shared_ptr<Matrix> last_hidden = LRN::inter_channel(input_matrix, LRN_dimensions[lrn_predict_idx][0], LRN_dimensions[lrn_predict_idx][1], 
																 LRN_dimensions[lrn_predict_idx][2], LRN_info[lrn_predict_idx][0], LRN_info[lrn_predict_idx][1], 
																 LRN_info[lrn_predict_idx][2], LRN_info[lrn_predict_idx][3]);
			lrn_idx += 1;
		}else if(inner_layers[0] == 2){

		}else if(inner_layers[0] == 3){

		}else if(inner_layers[0] == 4){

		}else if(inner_layers[0] == 5){

		}
	}


	if(inner_layers[inner_layers.size() - 1] == 0){
		std::shared_ptr<Matrix> output = fully_connected::feed_forward(last_hidden, weights[0], biases[0], fully_connected_activations[0]);
	}else if(inner_layers[0] == 1){
		std::shared_ptr<Matrix> output = LRN::inter_channel(input_matrix, LRN_dimensions[lrn_predict_idx][0], LRN_dimensions[lrn_predict_idx][1], 
															 LRN_dimensions[lrn_predict_idx][2], LRN_info[lrn_predict_idx][0], LRN_info[lrn_predict_idx][1], 
															 LRN_info[lrn_predict_idx][2], LRN_info[lrn_predict_idx][3]);
		lrn_idx += 1;
	}else if(inner_layers[0] == 2){

	}else if(inner_layers[0] == 3){

	}else if(inner_layers[0] == 4){

	}else if(inner_layers[0] == 5){

	}



}

#endif
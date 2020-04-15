#ifndef PREDICT_INCLUDE
#define PREDICT_INCLUDE


std::shared_ptr<Matrix> NeuralNetwork::Predict(const Matrix& input_array, const bool vectorize_inputs = true){
	if(vectorize_inputs){
		std::vector<int> inp_shape = input_array.shape; inp_shape.push_back(1);
		Matrix input_matrix = Matrix(inp_shape); input_matrix.matrix_values = input_array.matrix_values;
	}else{
		Matrix& input_matrix = input_array;
	}

	int lrn_predict_idx = 0; // keeping track of how many times the next layer has been an lrn layer so that the dims (channels, width, height) 
	//and the info (epsilon, radius, etc. ) can be accessed
	int fcl_predict_idx = 0; //same idea with other layers
	int cnv_predict_idx = 0;
	int mxp_predict_idx = 0;



	if(inner_layers[0] == 0){
		std::shared_ptr<Matrix> last_hidden = fully_connected::feed_forward(input_matrix, weights[0], biases[0], fully_connected_activations[fcl_predict_idx]);
		fcl_predict_idx++;
	}else if(inner_layers[0] == 1){
		std::shared_ptr<Matrix> last_hidden = LRN::inter_channel(input_matrix, LRN_dimensions[lrn_predict_idx][0], LRN_dimensions[lrn_predict_idx][1], 
															 LRN_dimensions[lrn_predict_idx][2], LRN_info[lrn_predict_idx][0], LRN_info[lrn_predict_idx][1], 
															 LRN_info[lrn_predict_idx][2], LRN_info[lrn_predict_idx][3]);
		lrn_predict_idx++;
	}else if(inner_layers[0] == 2){
		std::shared_ptr<Matrix> last_hidden = LRN::intra_channel(input_matrix, LRN_dimensions[lrn_predict_idx][0], LRN_dimensions[lrn_predict_idx][1], 
															 LRN_dimensions[lrn_predict_idx][2], LRN_info[lrn_predict_idx][0], LRN_info[lrn_predict_idx][1], 
															 LRN_info[lrn_predict_idx][2], LRN_info[lrn_predict_idx][3]);
		lrn_predict_idx++;
	}else if(inner_layers[0] == 3){
		std::shared_ptr<Matrix> last_hidden = input_matrix;
	}else if(inner_layers[0] == 4){
		std::shared_ptr<Matrix> last_hidden = CNV::convolution(input_matrix, conv_info_array[cnv_predict_idx][5], conv_info_array[cnv_predict_idx][0], conv_info_array[cnv_predict_idx][1],
															   weights[0], biases[0], conv_info_array[cnv_predict_idx][4], conv_info_array[cnv_predict_idx][2]);
		cnv_predict_idx++;
	}else if(inner_layers[0] == 5){
		std::shared_ptr<Matrix> last_hidden = CNV::maxpool(input_matrix, maxpool_info_array[2], maxpool_info_array[3], maxpool_info_array[4], maxpool_info_array[0], maxpool_info_array[1]);
		mxp_predict_idx++;
	}

	std::vector<Matrix> hiddens = {last_hidden};
	for(int l = 0; l < inner_layers.size(); l++){
		if(inner_layers[0] == 0){
			std::shared_ptr<Matrix> last_hidden = fully_connected::feed_forward(hiddens[hiddens.size() - 1], weights[l + 1], biases[l + 1], fully_connected_activations[fcl_predict_idx]);
			fcl_predict_idx++;
		}else if(inner_layers[0] == 1){
			std::shared_ptr<Matrix> last_hidden = LRN::inter_channel(hiddens[hiddens.size() - 1], LRN_dimensions[lrn_predict_idx][0], LRN_dimensions[lrn_predict_idx][1], 
																 LRN_dimensions[lrn_predict_idx][2], LRN_info[lrn_predict_idx][0], LRN_info[lrn_predict_idx][1], 
																 LRN_info[lrn_predict_idx][2], LRN_info[lrn_predict_idx][3]);
			lrn_predict_idx++;
		}else if(inner_layers[0] == 2){
			std::shared_ptr<Matrix> last_hidden = LRN::intra_channel(hiddens[hiddens.size() - 1], LRN_dimensions[lrn_predict_idx][0], LRN_dimensions[lrn_predict_idx][1], 
																 LRN_dimensions[lrn_predict_idx][2], LRN_info[lrn_predict_idx][0], LRN_info[lrn_predict_idx][1], 
																 LRN_info[lrn_predict_idx][2], LRN_info[lrn_predict_idx][3]);
			lrn_predict_idx++;
		}else if(inner_layers[0] == 3){
			std::shared_ptr<Matrix> last_hidden = hiddens[hiddens.size() - 1];
		}else if(inner_layers[0] == 4){
			std::shared_ptr<Matrix> last_hidden = CNV::convolution(hiddens[hiddens.size() - 1], conv_info_array[cnv_predict_idx][5], conv_info_array[cnv_predict_idx][0], conv_info_array[cnv_predict_idx][1],
																   weights[l + 1], biases[l + 1], conv_info_array[cnv_predict_idx][4], conv_info_array[cnv_predict_idx][2]);
			cnv_predict_idx++;
		}else if(inner_layers[0] == 5){
			std::shared_ptr<Matrix> last_hidden = CNV::maxpool(hiddens[hiddens.size() - 1], maxpool_info_array[2], maxpool_info_array[3], maxpool_info_array[4], maxpool_info_array[0], maxpool_info_array[1]);
			mxp_predict_idx++;
		}

		hiddens.push_back(last_hidden);
	}

	if(inner_layers[0] == 0){
		std::shared_ptr<Matrix> output = fully_connected::feed_forward(hiddens[hiddens.size() - 1], weights[weights.size() - 1], biases[biases.size() - 1], fully_connected_activations[fcl_predict_idx]);
		fcl_predict_idx++;
	}else if(inner_layers[0] == 1){
		std::shared_ptr<Matrix> output = LRN::inter_channel(hiddens[hiddens.size() - 1], LRN_dimensions[lrn_predict_idx][0], LRN_dimensions[lrn_predict_idx][1], 
															 LRN_dimensions[lrn_predict_idx][2], LRN_info[lrn_predict_idx][0], LRN_info[lrn_predict_idx][1], 
															 LRN_info[lrn_predict_idx][2], LRN_info[lrn_predict_idx][3]);
		lrn_predict_idx++;
	}else if(inner_layers[0] == 2){
		std::shared_ptr<Matrix> output = LRN::intra_channel(hiddens[hiddens.size() - 1], LRN_dimensions[lrn_predict_idx][0], LRN_dimensions[lrn_predict_idx][1], 
															 LRN_dimensions[lrn_predict_idx][2], LRN_info[lrn_predict_idx][0], LRN_info[lrn_predict_idx][1], 
															 LRN_info[lrn_predict_idx][2], LRN_info[lrn_predict_idx][3]);
		lrn_predict_idx++;
	}else if(inner_layers[0] == 3){
		std::shared_ptr<Matrix> output = hiddens[hiddens.size() - 1];
	}else if(inner_layers[0] == 4){
		std::shared_ptr<Matrix> output = CNV::convolution(hiddens[hiddens.size() - 1], conv_info_array[cnv_predict_idx][5], conv_info_array[cnv_predict_idx][0], conv_info_array[cnv_predict_idx][1],
															   weights[weights.size() - 1], biases[biases.size() - 1], conv_info_array[cnv_predict_idx][4], conv_info_array[cnv_predict_idx][2]);
		cnv_predict_idx++;
	}else if(inner_layers[0] == 5){
		std::shared_ptr<Matrix> output = CNV::maxpool(hiddens[hiddens.size() - 1], maxpool_info_array[2], maxpool_info_array[3], maxpool_info_array[4], maxpool_info_array[0], maxpool_info_array[1]);
		mxp_predict_idx++;
	}
	
	return output;
}

#endif
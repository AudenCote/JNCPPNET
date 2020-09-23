#ifndef PREDICT_INCLUDE
#define PREDICT_INCLUDE


std::vector<std::shared_ptr<Matrix>> NeuralNetwork::feed_forward_all_template(const Matrix& input_array, const bool vectorize_inputs = true) {

	std::vector<int> inp_shape;
	if (vectorize_inputs) {
		inp_shape = input_array.shape; inp_shape.push_back(1);
	}
	else {
		inp_shape = input_array.shape;
	}
	Matrix input_matrix = Matrix(inp_shape); input_matrix.matrix_values = input_array.matrix_values;

	int lrn_predict_idx = 0; // keeping track of how many times the next layer has been an lrn layer so that the dims (channels, width, height) 
	//and the info (epsilon, radius, etc. ) can be accessed from the lrn_info matrix
	int fcl_predict_idx = 0; //same idea with other layers
	int cnv_predict_idx = 0;
	int mxp_predict_idx = 0;
	int avgp_predict_idx = 0;
	int glob_avgp_predict_idx = 0;


	//Checking the first layer type in the inner layers matrix (see structure - private declarations)
	std::shared_ptr<Matrix> last_hidden;
	if (inner_layers[0] == 0) {
		last_hidden = fully_connected::feed_forward(input_matrix, *weights[0], *biases[0], fully_connected_activations[fcl_predict_idx]);
		fcl_predict_idx++;
	}
	else if (inner_layers[0] == 1) {
		last_hidden = LRN::inter_channel(input_matrix, LRN_dimensions[lrn_predict_idx][0], LRN_dimensions[lrn_predict_idx][1],
			LRN_dimensions[lrn_predict_idx][2], LRN_info[lrn_predict_idx][0], LRN_info[lrn_predict_idx][1],
			LRN_info[lrn_predict_idx][2], LRN_info[lrn_predict_idx][3]);
		Matrix::Reshape(*last_hidden, { input_matrix.shape[0], 1 });
		lrn_predict_idx++;
	}
	else if (inner_layers[0] == 2) {
		last_hidden = LRN::intra_channel(input_matrix, LRN_dimensions[lrn_predict_idx][0], LRN_dimensions[lrn_predict_idx][1],
			LRN_dimensions[lrn_predict_idx][2], LRN_info[lrn_predict_idx][0], LRN_info[lrn_predict_idx][1],
			LRN_info[lrn_predict_idx][2], LRN_info[lrn_predict_idx][3]);
		Matrix::Reshape(*last_hidden, { input_matrix.shape[0], 1 });
		lrn_predict_idx++;
	}
	else if (inner_layers[0] == 3) {
		*last_hidden = input_matrix;
	}
	else if (inner_layers[0] == 4) {
		last_hidden = CNV::convolution(input_matrix, conv_info_array[cnv_predict_idx][5], conv_info_array[cnv_predict_idx][0], conv_info_array[cnv_predict_idx][1],
			*weights[0], *biases[0], conv_info_array[cnv_predict_idx][4], conv_info_array[cnv_predict_idx][2], conv_activations[cnv_predict_idx]);
		Matrix::Reshape(*last_hidden, { conv_info_array[cnv_predict_idx][5] * ((int)((input_matrix.shape[1] - conv_info_array[cnv_predict_idx][2]) / conv_info_array[cnv_predict_idx][4]) + 1) *
			((int)((input_matrix.shape[1] - conv_info_array[cnv_predict_idx][2]) / conv_info_array[cnv_predict_idx][4]) + 1), 1 });
		cnv_predict_idx++;
		cnv_predict_idx++;
	}
	else if (inner_layers[0] == 5) {
		last_hidden = CNV::maxpool(input_matrix, maxpool_info_array[mxp_predict_idx][2], maxpool_info_array[mxp_predict_idx][3], maxpool_info_array[mxp_predict_idx][4], maxpool_info_array[mxp_predict_idx][0], maxpool_info_array[mxp_predict_idx][1]);
		Matrix::Reshape(*last_hidden, { maxpool_info_array[mxp_predict_idx][2] * maxpool_info_array[mxp_predict_idx][3] * maxpool_info_array[mxp_predict_idx][4], 1 });
		mxp_predict_idx++;
	}
	else if (inner_layers[0] == 6) {
		last_hidden = CNV::avgpool(input_matrix, avgpool_info_array[mxp_predict_idx][2], avgpool_info_array[mxp_predict_idx][3], avgpool_info_array[mxp_predict_idx][4], avgpool_info_array[mxp_predict_idx][0], avgpool_info_array[mxp_predict_idx][1]);
		Matrix::Reshape(*last_hidden, { avgpool_info_array[avgp_predict_idx][2] * avgpool_info_array[avgp_predict_idx][3] * avgpool_info_array[avgp_predict_idx][4] , 1 });
		avgp_predict_idx++;
	}
	else if (inner_layers[0] == 7) {
		last_hidden = CNV::global_avgpool(input_matrix, globavgpool_info_array[glob_avgp_predict_idx][0], globavgpool_info_array[glob_avgp_predict_idx][2], globavgpool_info_array[glob_avgp_predict_idx][1]);
		Matrix::Reshape(*last_hidden, { globavgpool_info_array[glob_avgp_predict_idx][0], 1 });
		glob_avgp_predict_idx++; 
	}

	std::vector<std::shared_ptr<Matrix>> hiddens = { last_hidden };
	for (int l = 0; l < inner_layers.size(); l++) {
		if (inner_layers[0] == 0) {
			last_hidden = fully_connected::feed_forward(*hiddens[hiddens.size() - 1], *weights[l + 1], *biases[l + 1], fully_connected_activations[fcl_predict_idx]);
			fcl_predict_idx++;
		}
		else if (inner_layers[0] == 1) {
			last_hidden = LRN::inter_channel(*hiddens[hiddens.size() - 1], LRN_dimensions[lrn_predict_idx][0], LRN_dimensions[lrn_predict_idx][1],
				LRN_dimensions[lrn_predict_idx][2], LRN_info[lrn_predict_idx][0], LRN_info[lrn_predict_idx][1],
				LRN_info[lrn_predict_idx][2], LRN_info[lrn_predict_idx][3]);
			Matrix::Reshape(*last_hidden, { hiddens[hiddens.size() - 1]->shape[0], 1 });
			lrn_predict_idx++;
		}
		else if (inner_layers[0] == 2) {
			last_hidden = LRN::intra_channel(*hiddens[hiddens.size() - 1], LRN_dimensions[lrn_predict_idx][0], LRN_dimensions[lrn_predict_idx][1],
				LRN_dimensions[lrn_predict_idx][2], LRN_info[lrn_predict_idx][0], LRN_info[lrn_predict_idx][1],
				LRN_info[lrn_predict_idx][2], LRN_info[lrn_predict_idx][3]);
			Matrix::Reshape(*last_hidden, { hiddens[hiddens.size() - 1]->shape[0], 1 });
			lrn_predict_idx++;
		}
		else if (inner_layers[0] == 3) {
			last_hidden = hiddens[hiddens.size() - 1];
		}
		else if (inner_layers[0] == 4) {
			last_hidden = CNV::convolution(*hiddens[hiddens.size() - 1], conv_info_array[cnv_predict_idx][5], conv_info_array[cnv_predict_idx][0], conv_info_array[cnv_predict_idx][1],
				*weights[l + 1], *biases[l + 1], conv_info_array[cnv_predict_idx][4], conv_info_array[cnv_predict_idx][2]);
			Matrix::Reshape(*last_hidden, { conv_info_array[cnv_predict_idx][5] * ((int)((hiddens[hiddens.size() - 1]->shape[1] - conv_info_array[cnv_predict_idx][2]) / conv_info_array[cnv_predict_idx][4]) + 1) *
			((int)((hiddens[hiddens.size() - 1]->shape[1] - conv_info_array[cnv_predict_idx][2]) / conv_info_array[cnv_predict_idx][4]) + 1), 1 });
			cnv_predict_idx++;
		}
		else if (inner_layers[0] == 5) {
			last_hidden = CNV::maxpool(*hiddens[hiddens.size() - 1], maxpool_info_array[mxp_predict_idx][2], maxpool_info_array[mxp_predict_idx][3], maxpool_info_array[mxp_predict_idx][4], maxpool_info_array[mxp_predict_idx][0], maxpool_info_array[mxp_predict_idx][1]);
			Matrix::Reshape(*last_hidden, { maxpool_info_array[mxp_predict_idx][2] * maxpool_info_array[mxp_predict_idx][3] * maxpool_info_array[mxp_predict_idx][4], 1 });
			mxp_predict_idx++;
		}
		else if (inner_layers[0] == 6) {
			last_hidden = CNV::avgpool(*hiddens[hiddens.size() - 1], avgpool_info_array[avgp_predict_idx][2], avgpool_info_array[avgp_predict_idx][3], avgpool_info_array[avgp_predict_idx][4], avgpool_info_array[avgp_predict_idx][0], avgpool_info_array[avgp_predict_idx][1]);
			Matrix::Reshape(*last_hidden, { avgpool_info_array[avgp_predict_idx][2] * avgpool_info_array[avgp_predict_idx][3] * avgpool_info_array[avgp_predict_idx][4] , 1 });
			avgp_predict_idx++;
		}
		else if (inner_layers[0] == 7) {
			last_hidden = CNV::global_avgpool(*hiddens[hiddens.size() - 1], globavgpool_info_array[glob_avgp_predict_idx][0], globavgpool_info_array[glob_avgp_predict_idx][2], globavgpool_info_array[glob_avgp_predict_idx][1]);
			Matrix::Reshape(*last_hidden, { globavgpool_info_array[glob_avgp_predict_idx][0], 1 });
			glob_avgp_predict_idx++;
		}

		hiddens.push_back(last_hidden);
	}

	std::shared_ptr<Matrix> output;

	if (inner_layers[0] == 0) {
		output = fully_connected::feed_forward(*hiddens[hiddens.size() - 1], *weights[weights.size() - 1], *biases[biases.size() - 1], fully_connected_activations[fcl_predict_idx]); //This should already be a vector
		fcl_predict_idx++;
	}
	else if (inner_layers[0] == 1) {
		output = LRN::inter_channel(*hiddens[hiddens.size() - 1], LRN_dimensions[lrn_predict_idx][0], LRN_dimensions[lrn_predict_idx][1],
			LRN_dimensions[lrn_predict_idx][2], LRN_info[lrn_predict_idx][0], LRN_info[lrn_predict_idx][1],
			LRN_info[lrn_predict_idx][2], LRN_info[lrn_predict_idx][3]);
		Matrix::Reshape(*output, { hiddens[hiddens.size() - 1]->shape[0], 1 });
		lrn_predict_idx++;
	}
	else if (inner_layers[0] == 2) {
		output = LRN::intra_channel(*hiddens[hiddens.size() - 1], LRN_dimensions[lrn_predict_idx][0], LRN_dimensions[lrn_predict_idx][1],
			LRN_dimensions[lrn_predict_idx][2], LRN_info[lrn_predict_idx][0], LRN_info[lrn_predict_idx][1],
			LRN_info[lrn_predict_idx][2], LRN_info[lrn_predict_idx][3]);
		Matrix::Reshape(*output, { hiddens[hiddens.size() - 1]->shape[0], 1 });
		lrn_predict_idx++;
	}
	else if (inner_layers[0] == 3) {
		output = hiddens[hiddens.size() - 1]; //This will always be a vector, needs no reshaping
	}
	else if (inner_layers[0] == 4) {
		output = CNV::convolution(*hiddens[hiddens.size() - 1], conv_info_array[cnv_predict_idx][5], conv_info_array[cnv_predict_idx][0], conv_info_array[cnv_predict_idx][1],
			*weights[weights.size() - 1], *biases[biases.size() - 1], conv_info_array[cnv_predict_idx][4], conv_info_array[cnv_predict_idx][2], conv_activations[cnv_predict_idx]);
		Matrix::Reshape(*output, { conv_info_array[cnv_predict_idx][5] * ((int)((hiddens[hiddens.size() - 1]->shape[1] - conv_info_array[cnv_predict_idx][2]) / conv_info_array[cnv_predict_idx][4]) + 1) *
			((int)((hiddens[hiddens.size() - 1]->shape[1] - conv_info_array[cnv_predict_idx][2]) / conv_info_array[cnv_predict_idx][4]) + 1), 1});
		cnv_predict_idx++;
	}
	else if (inner_layers[0] == 5) {
		output = CNV::maxpool(*hiddens[hiddens.size() - 1], maxpool_info_array[mxp_predict_idx][2], maxpool_info_array[mxp_predict_idx][3], maxpool_info_array[mxp_predict_idx][4], maxpool_info_array[mxp_predict_idx][0], maxpool_info_array[mxp_predict_idx][1]);
		Matrix::Reshape(*output, { maxpool_info_array[mxp_predict_idx][2] * maxpool_info_array[mxp_predict_idx][3] * maxpool_info_array[mxp_predict_idx][4], 1});
		mxp_predict_idx++;
	}
	else if (inner_layers[0] == 6) {
		output = CNV::avgpool(*hiddens[hiddens.size() - 1], avgpool_info_array[avgp_predict_idx][2], avgpool_info_array[avgp_predict_idx][3], avgpool_info_array[avgp_predict_idx][4], avgpool_info_array[avgp_predict_idx][0], avgpool_info_array[avgp_predict_idx][1]);
		Matrix::Reshape(*output, { avgpool_info_array[avgp_predict_idx][2] * avgpool_info_array[avgp_predict_idx][3] * avgpool_info_array[avgp_predict_idx][4] , 1 });
		avgp_predict_idx++;
	}
	else if (inner_layers[0] == 7) {	
		output = CNV::global_avgpool(*hiddens[hiddens.size() - 1], globavgpool_info_array[glob_avgp_predict_idx][0], globavgpool_info_array[glob_avgp_predict_idx][1], globavgpool_info_array[glob_avgp_predict_idx][2]);
		Matrix::Reshape(*output, { globavgpool_info_array[glob_avgp_predict_idx][0], 1 });
		glob_avgp_predict_idx++;
	}

	//Putting into the return vector all of the data that needs to be accessed during training
	std::vector<std::shared_ptr<Matrix>> return_vector = { output };
	for (std::shared_ptr<Matrix> hidden : hiddens) { return_vector.push_back(hidden); }
	for (std::shared_ptr<Matrix> weight_m : weights) { return_vector.push_back(weight_m); }
	for (std::shared_ptr<Matrix> bias_v : biases) { return_vector.push_back(bias_v); }
	return return_vector;
}


std::shared_ptr<Matrix> NeuralNetwork::Predict(const Matrix& input_array, const bool vectorize_inputs = true) {
	std::vector<std::shared_ptr<Matrix>> feed_forward_return_vector = feed_forward_all_template(input_array, vectorize_inputs);
	return feed_forward_return_vector[0];
}


#endif
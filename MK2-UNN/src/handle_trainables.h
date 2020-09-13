#ifndef HANDLE_TRAINABLES_INCLUDE
#define HANDLE_TRAINABLES_INCLUDE

//input layer -> -1
//output layer -> -2
//fully connected layer -> 0
//inter_channel LRN -> 1
//intra_channel LRN -> 2
//batch_norm -> 3
//convolutional layer -> 4
//max-pooling layer -> 5

int NeuralNetwork::get_norm_layer_outputs(const int layer_index) {
	int last_relevant = -5;
	//might have to start at layer_index - 1? doing this convolutedly so that it can be error-checked, can be reworked
	for (int i = layer_index; i >= 0; i++) {
		if (inner_layers[i] != 1 && inner_layers[i] != 2) {
			last_relevant = inner_layers[i];
			break;
		}
	}

	if (last_relevant == 4) {
		return conv_nodes_array[cnv_idx - 1];
	}
	else if (last_relevant == 5) {
		return conv_nodes_array[mxp_idx - 1];
	}
	else if (last_relevant == 0) {
		Logger::Warning("Local response normalization layers are designed primarily to come after a convolutional-type layer");
		return fully_connected_nodes_array[fcl_idx - 1];
	}
	else {
		return -5;
	}

}

void NeuralNetwork::handle_trainables(const int layer_index, const int prev_i, const int i) {
	try {
		if (i == -2) {
			std::vector<int> w_s;
			if (prev_i == -1) {
				w_s = { output_nodes, input_nodes };
			}
			else if (prev_i == 0) {
				w_s = { output_nodes, fully_connected_nodes_array[fcl_idx - 1] };
			}
			else if (prev_i == 4) {
				w_s = { output_nodes, conv_nodes_array[cnv_idx - 1] };
			}
			else if (prev_i == 5) {
				w_s = { output_nodes, maxpool_nodes_array[mxp_idx - 1] };
			}
			else if (prev_i == 1 || prev_i == 2) {
				if (get_norm_layer_outputs(layer_index) != -5) {
					w_s = { output_nodes, get_norm_layer_outputs(layer_index) };
				}
				else {
					throw(std::logic_error("No valid previous layers to check for output nodes. Should have hit at least input layer -- logic error\nException thrown in function NeuralNetwork::handle_trainables()"));
				}
			}
			else if (prev_i == 3) {
				if (get_norm_layer_outputs(layer_index) != -5) {
					w_s = { output_nodes, get_norm_layer_outputs(layer_index) };
				}
				else {
					throw(std::logic_error("No valid previous layers to check for output nodes. Should have hit at least input layer -- logic error\nException thrown in function NeuralNetwork::handle_trainables()"));
				}
			}
			std::vector<int> b_s = { output_nodes, 1 };
			weights.push_back(std::make_shared<Matrix>(w_s));
			biases.push_back(std::make_shared<Matrix>(b_s));
		}
		else if (i == 0) {
			std::vector<int> w_s;
			if (prev_i == -1) {
				w_s = { fully_connected_nodes_array[fcl_idx], input_nodes };
			}
			else if (prev_i == 4) {
				w_s = { fully_connected_nodes_array[fcl_idx], conv_nodes_array[cnv_idx - 1] };
			}
			else if (prev_i == 0) {
				w_s = { fully_connected_nodes_array[fcl_idx], fully_connected_nodes_array[fcl_idx - 1] };
			}
			else if (prev_i == 1 || prev_i == 2) {
				if (get_norm_layer_outputs(layer_index) != -5) {
					w_s = { output_nodes, get_norm_layer_outputs(layer_index) };
				}
				else {
					throw(std::logic_error("No valid previous layers to check for output nodes. Should have hit at least input layer -- logic error\nException thrown in function NeuralNetwork::handle_trainables()"));
				}
			}
			else if (prev_i == 3) {
				if (get_norm_layer_outputs(layer_index) != -5) {
					w_s = { output_nodes, get_norm_layer_outputs(layer_index) };
				}
				else {
					throw(std::logic_error("No valid previous layers to check for output nodes. Should have hit input layer -- logic error\nException thrown in function NeuralNetwork::handle_trainables()"));
				}
			}
			else if (prev_i == 5) {
				w_s = { fully_connected_nodes_array[fcl_idx], maxpool_nodes_array[mxp_idx - 1] };
			}
			std::vector<int> b_s = { fully_connected_nodes_array[fcl_idx], 1 };
			weights.push_back(std::make_shared<Matrix>(w_s));
			biases.push_back(std::make_shared<Matrix>(b_s));
			fcl_idx += 1;
		}
		else if (i == 1 || i == 2) {
			if (prev_i != 4 && prev_i != 5) {
				Logger::Warning("Local response normalization layers are designed primarily to come after a convolutional-type layer");
			}
			std::vector<int> w_s;
			if (prev_i == -1) {
				w_s = { input_nodes, input_nodes };
				std::vector<int> b_s = { input_nodes, 1 };
			}
			else if (prev_i == 0) {
				w_s = { fully_connected_nodes_array[fcl_idx - 1], fully_connected_nodes_array[fcl_idx - 1] };
				std::vector<int> b_s = { fully_connected_nodes_array[fcl_idx - 1], 1 };
			}
			//Make sure this is correct
			else if (prev_i == 1 || prev_i == 2) {
				int glo = get_norm_layer_outputs(layer_index);
				if (glo != -5) {
					w_s = { glo, glo };
					std::vector<int> b_s = { glo, 1 };
				}
				else {
					throw(std::logic_error("No valid previous layers to check for output nodes. Should have hit at least input layer -- logic error\nException thrown in function NeuralNetwork::handle_trainables()"));
				}
			}
			else if (prev_i == 3) {
				int glo = get_norm_layer_outputs(layer_index);
				if (glo != -5) {
					w_s = { glo, glo };
					std::vector<int> b_s = { glo, 1 };
				}
				else {
					throw(std::logic_error("No valid previous layers to check for output nodes. Should have hit at least input layer -- logic error\nException thrown in function NeuralNetwork::handle_trainables()"));
				}
			}
			else if (prev_i == 4) {
				w_s = { conv_nodes_array[cnv_idx - 1], conv_nodes_array[cnv_idx - 1] };
				std::vector<int> b_s = { conv_nodes_array[cnv_idx - 1], 1 };
			}
			else if (prev_i == 5) {
				w_s = { maxpool_nodes_array[mxp_idx - 1], maxpool_nodes_array[mxp_idx - 1] };
				std::vector<int> b_s = { maxpool_nodes_array[mxp_idx - 1], 1 };
			}

			if (i == 1) {
				lrn_inter_idx += 1;
			}
			else if (i == 2) {
				lrn_intra_idx += 1;
			}
		}
		else if (i == 3) {
			std::vector<int> bnt_s;
			if (prev_i == -1) {
				Logger::Warning("Batch Normalization is not intended to be used on the input layer - consider randomizing inputs, etc.\nNeural network being structured in accordance with the users preferences");
				bnt_s = { input_nodes, 1 };
			}
			else if (prev_i == 0) {
				bnt_s = { fully_connected_nodes_array[fcl_idx - 1], 1 };
			}
			else if (prev_i == 1 || prev_i == 2 || prev_i == 3) {
				if (prev_i != 3) Logger::Warning("Batch Normalization is not intended to be used in conjunction with local response normalization\nNeural network being structured in accordance with the users preferences");
				int glo = get_norm_layer_outputs(layer_index);
				if (glo != -5) {
					bnt_s = { glo, 1 };
				}
				else {
					throw(std::logic_error("No valid previous layers to check for output nodes. Should have hit at least input layer -- logic error\nException thrown in function NeuralNetwork::handle_trainables()"));
				}
			}
			else if (prev_i == 4) {
				bnt_s = { conv_nodes_array[cnv_idx - 1], 1 };
			}
			else if (prev_i == 5) {
				bnt_s = { maxpool_nodes_array[mxp_idx - 1], 1 };
			}

			bnt_inner_shapes.push_back(bnt_s);
			std::vector<float> trainables = { my_misc_utils::gen_random_float(-1, 1), my_misc_utils::gen_random_float(-1, 1) }; bnt_trainables.push_back(trainables);
			bnt_idx += 1;
		}
		else if (i == 4) {
			std::vector<int> filters_s = { conv_info_array[cnv_idx - 1][3], conv_info_array[cnv_idx - 1][5], conv_info_array[cnv_idx][2], conv_info_array[cnv_idx][2] };
			if (prev_i == -1) {
				int pro = 1;
				for (int v : filters_s) pro = pro * v;
				if (pro != input_nodes) throw(std::logic_error("Image dimensions do not match previous layer node count -- logic error thrown in function NeuralNetwork::handle_trainables() while constructing convolutional layer"));
			}
			else if (prev_i == 0) {
				int pro = 1;
				for (int v : filters_s) pro = pro * v;
				if (pro != fully_connected_nodes_array[fcl_idx - 1]) throw(std::logic_error("Image dimensions do not match previous layer node count -- logic error thrown in function NeuralNetwork::handle_trainables() while constructing convolutional layer"));
			}
			else if (prev_i == 4) {
				int pro = 1;
				for (int v : filters_s) pro = pro * v;
				if (pro != conv_nodes_array[cnv_idx - 1]) throw(std::logic_error("Image dimensions do not match previous layer node count -- logic error thrown in function NeuralNetwork::handle_trainables() while constructing convolutional layer"));
			}
			else if (prev_i == 5) {
				int pro = 1;
				for (int v : filters_s) pro = pro * v;
				if (pro != maxpool_nodes_array[mxp_idx - 1]) throw(std::logic_error("Image dimensions do not match previous layer node count -- logic error thrown in function NeuralNetwork::handle_trainables() while constructing convolutional layer"));
			}
			else if (prev_i == 1 || prev_i == 2) {
				int glo = get_norm_layer_outputs(layer_index);
				if (glo != -5) {
					int pro = 1;
					for (int v : filters_s) pro = pro * v;
					if (pro != glo) throw(std::logic_error("Image dimensions do not match previous layer node count -- logic error thrown in function NeuralNetwork::handle_trainables() while constructing convolutional layer"));
				}
				else {
					throw(std::logic_error("No valid previous layers to check for output nodes. Should have hit at least input layer -- logic error\nException thrown in function NeuralNetwork::handle_trainables()"));
				}
			}
			else if (prev_i == 3) {
				int glo = get_norm_layer_outputs(layer_index);
				if (glo != -5) {
					int pro = 1;
					for (int v : filters_s) pro = pro * v;
					if (pro != glo) throw(std::logic_error("Image dimensions do not match previous layer node count -- logic error thrown in function NeuralNetwork::handle_trainables() while constructing convolutional layer"));
				}
				else {
					throw(std::logic_error("No valid previous layers to check for output nodes. Should have hit at least input layer -- logic error\nException thrown in function NeuralNetwork::handle_trainables()"));
				}
			}
			std::vector<int> b_s = { conv_info_array[cnv_idx][3], 1 }; //one bias for each filter - same over channels, to put emphasis on features
			weights.push_back(std::make_shared<Matrix>(filters_s));
			biases.push_back(std::make_shared<Matrix>(b_s));
			cnv_idx += 1;
		}
		else if (i == 5) {
			//max-pooling layers use no trainable parameters, but still need to increment the count to access layer information when iterating through other layers
			mxp_idx += 1;
		}
		else if (i == 6) {
			avgp_idx += 1;
		}
		else if (i == 7) {
		globavgp_idx += 1;
		}
	}
	catch (const std::logic_error& e) {
		Logger::Error(e.what());
	}
}

#endif
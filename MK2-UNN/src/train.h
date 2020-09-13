#ifndef train_include
#define train_include

void NeuralNetwork::Train(Matrix& training_data, Matrix& target_data, int epochs, int batch_size, float learning_rate, const char* gradient_descent_type = "mini-batch", bool print = true) {
	if (training_data.shape[0] != target_data.shape[0])
		std::cout << "Different Sample Lengths" << std::endl;

	std::vector<int> input_batches_shape = { (int)ceil(training_data.shape[0] / batch_size), batch_size, input_nodes, 1 }; Matrix input_batches(input_batches_shape);
	for (int i = 0; i < training_data.shape[0]; ++i) {
		if (i % batch_size == 0) {
			for (int j = 0; j < batch_size; ++j) {
				for (int k = 0; k < input_nodes; ++k) {
					input_batches.SetVal({ (int)ceil(i / batch_size), j, k, 0 }, (float)training_data.matrix_values[i + (j * input_nodes) + k]);
				}
			}
		}
	}

	std::vector<int> target_batches_shape = { (int)ceil(target_data.shape[0] / batch_size), batch_size, output_nodes, 1 }; Matrix target_batches(target_batches_shape);
	for (int i = 0; i < target_data.shape[0]; ++i) {
		if (i % batch_size == 0)
			for (int j = 0; j < batch_size; ++j) {
				for (int k = 0; k < output_nodes; ++k) {
					target_batches.SetVal({ (int)ceil(i / batch_size), j, k, 0 }, (float)target_data.matrix_values[i + (j * output_nodes) + k]);
				}
			}
	}

	std::vector<double> losses;
	for (int e = 0; e < epochs; e++) {

		for (int b = 0; b < input_batches.shape[0]; b++) {

			std::shared_ptr<Matrix> input_batch = input_batches.GetChunk({ b });
			std::shared_ptr<Matrix> target_batch = target_batches.GetChunk({ b });

			std::vector<std::vector<std::shared_ptr<Matrix>>> fcl_bias_deltas;
			std::vector<std::vector<std::shared_ptr<Matrix>>> fcl_weights_deltas;
			std::vector<std::vector<std::shared_ptr<Matrix>>> cnv_bias_deltas;
			std::vector<std::vector<std::shared_ptr<Matrix>>> cnv_filt_deltas;

			double loss;

			for (int p = 0; p < batch_size; p++) {
				std::vector<std::shared_ptr<Matrix>> fcl_sample_weights_deltas; std::vector<std::shared_ptr<Matrix>> fcl_sample_bias_deltas; //Undo this nonsense of separating fcl and conv layer weights/filters: they are all in the same vector
				std::vector<std::shared_ptr<Matrix>> cnv_sample_filt_deltas; std::vector<std::shared_ptr<Matrix>> cnv_sample_bias_deltas;

				//==================================================================//
				//																	//
				//						FEED FORWARD 								//
				//																	//
				//==================================================================//


				std::shared_ptr<Matrix> inputs = input_batch->GetChunk({ p });
				std::shared_ptr<Matrix> targets = target_batch->GetChunk({ p });

				std::vector<std::shared_ptr<Matrix>> feed_forward_return_vector = feed_forward_all_template(*inputs, false);
				std::shared_ptr<Matrix>& outputs = feed_forward_return_vector[0];
				std::vector<std::shared_ptr<Matrix>> hiddens;
				for (int l = 1; l < inner_layers.size(); l++) {
					hiddens.push_back(feed_forward_return_vector[l]);
				}

				//==================================================================//
				//																	//
				//						BACK PROPOGATION 							//
				//																	//
				//==================================================================//

				//                  ===== OUTPUT LAYER =====

				if (output_layer_activation == "sigmoid" || output_layer_activation == "Sigmoid") {
					loss = Matrix::MeanSquareError(*targets, *outputs);
				}
				else if (output_layer_activation == "softmax" || output_layer_activation == "Softmax") {
					loss = Matrix::CategoricalCrossEntropy(*targets, *outputs);
				}

				std::shared_ptr<Matrix> errors = Matrix::ElementwiseSubtraction(*targets, *outputs);
				Matrix::SigmoidPrime(outputs);
				std::shared_ptr<Matrix> gradients = Matrix::ElementwiseMultiplication(*outputs, *errors);
				gradients->Multiply(learning_rate);

				std::shared_ptr<Matrix> hidden3_tr = Matrix::Transpose(hiddens[hiddens.size() - 1]);
				std::shared_ptr<Matrix> weights_ho_deltas = Matrix::DotProduct(*gradients, *hidden3_tr);

				sample_weights_deltas.push_back(weights_ho_deltas); //See above comment if this gives trouble
				sample_bias_deltas.push_back(gradients);

				//					 ===== INNER LAYERS: By definition these may not be -1 (input) or -2 (output) layers =====

				int weights_backprop_idx = weights.size() - 1;
				int cnv_info_backprop_idx = conv_info_array.size() - 1; int mxp_info_backprop_idx = maxpool_info_array.size() - 1; //To access the stride for each layer
				int fc_activation_idx = fully_connected_activations.size() - 1;
				int hiddens_idx = hiddens.size() - 1;
				
				//Vector to store the latest gradient produced in the inner layers iteration process
				std::shared_ptr<Matrix> last_gradients = errors;

				for (int l = inner_layers.size() - 1; l >= 0; l--) {
					if (inner_layers[l] == 0) {
						std::vector<std::shared_ptr<Matrix>> delt_grad_vec = fully_connected::backprop(last_gradients, weights[weights_backprop_idx], 
																												 hiddens[hiddens_idx], hiddens[hiddens_idx - 1], 
																												 learning_rate, fully_connected_activations[fc_activation_idx]);

						fcl_sample_weights_deltas.push_back(delt_grad_vec[0]); fcl_sample_bias_deltas.push_back(delt_grad_vec[1]);
						int gradients_vector_length = delt_grad_vec[1][0] * delt_grad_vec[1][1];
						last_gradients = Matrix::Reshape(*delt_grad_vec[1], {1, gradients_vector_length}); //Turning gradients into vector to be passed back 
						weights_backprop_idx -= 1; fc_activation_idx -= 1; hiddens_idx -= 1;
					}
					else if (inner_layers[l] == 1) {
						
					}
					else if (inner_layers[l] == 2) {



					}
					else if (inner_layers[l] == 3) {

					}
					else if (inner_layers[l] == 4) {
						std::vector<std::shared_ptr<Matrix>> conv_grads_vec = CNV::convolution_backprop(*last_gradients, *hiddens[hiddens_idx - 1], *weights[weights_backprop_idx], conv_info_array[cnv_info_backprop_idx][4]); //returns { dout, dfilt, dbias }

						cnv_sample_filt_deltas.push_back(conv_grads_vec[1]); cnv_sample_bias_deltas.push_back(conv_grads_vec[2]);
						int gradients_vector_length = conv_grads_vec[0][0] * conv_grads_vec[0][1] * conv_grads_vec[0][2];
						last_gradients = Matrix::Reshape(*conv_grads_vec[0], { 1, gradients_vector_length }); //Turning gradients into vector to be passed back 
						weights_backprop_idx -= 1; cnv_info_backprop_idx -= 1; hiddens_idx -= 1;
					}
					else if (inner_layers[l] == 5) {
						std::shared_ptr<Matrix> dpool = CNV::maxpool_backprop(); //Returns only dout, not a vector like some of the other layers (e.g. conv, fc)

						mxp_info_backprop_idx -= 1; hiddens_idx -= 1;
					}
					else if (inner_layers[l] == 6) {

					}
					else if (inner_layers[l] == 7) {

					}
				}

				//					===== Input Layer =====

				if (inner_layers[inner_layers.size() - 1] == 0) {
					std::vector<std::shared_ptr<Matrix>> delt_grad_vec = fully_connected::backprop(last_gradients, weights[weights_backprop_idx],
						hiddens[hiddens_idx], hiddens[hiddens_idx - 1],
						learning_rate, fully_connected_activations[fc_activation_idx]);
					sample_weights_deltas.push_back(delt_grad_vec[0]); sample_bias_deltas.push_back(delt_grad_vec[1]);
					weights_backprop_idx -= 1; fc_activation_idx -= 1; hiddens_idx -= 1;
				}
				else if (inner_layers[0] == 1) {

				}
				else if (inner_layers[0] == 2) {

				}
				else if (inner_layers[0] == 3) {

				}
				else if (inner_layers[0] == 4) {

				}
				else if (inner_layers[0] == 5) {

				}
				else if (inner_layers[0] == 6) {

				}
				else if (inner_layers[0] == 7) {

				}



			}


			if (print) {
				std::cout << "\n" << "================" << std::endl;
				std::cout << "Epoch: " << e + 1 << std::endl;
				std::cout << "Number of Epochs: " << epochs << std::endl;
				std::cout << "Batch: " << b + 1 << std::endl;
				std::cout << "Number of Batches: " << ceil(target_data.shape[0] / batch_size) << std::endl;
				std::cout << "Batch Size: " << batch_size << std::endl;
				std::cout << "Loss: " << loss << std::endl;
				std::cout << "================" << "\n" << std::endl;
			}

		}

	}

	std::vector<std::shared_ptr<Matrix>>& feed_forward_return_vector = feed_forward_all_template();

}

























#endif
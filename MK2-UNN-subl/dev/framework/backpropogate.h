#ifndef backpropogate_include
#define backpropogate_include


void NeuralNetwork::Train(std::shared_ptr<Matrix> training_data, std::shared_ptr<Matrix> target_data, const char* gradient_descent_type = "mini-batch", int epochs, int batch_size, bool print = true){
	if(training_data.shape[0] != target_data.shape[0])
			std::cout << "Different Sample Lengths" << std::endl;

		std::vector<int> input_batches_shape = {(int)ceil(training_data.shape[0]/batch_size), batch_size, input_nodes, 1}; Matrix input_batches(input_batches_shape);
		for(int i = 0; i < training_data.shape[0]; ++i){
			if(i % batch_size == 0){
				for(int j = 0; j < batch_size; ++j){
					for(int k = 0; k < input_nodes; ++k){
						input_batches.Set({(int)ceil(i/batch_size), j, k, 0}, (float)training_data.matrix_values[i + (j*input_nodes) + k]);
					}
				}
			}
		}
		
		std::vector<int> target_batches_shape = {(int)ceil(target_data.shape[0]/batch_size), batch_size, output_nodes, 1}; Matrix target_batches(target_batches_shape);
		for(int i = 0; i < target_data.shape[0]; ++i){
			if(i % batch_size == 0)
				for(int j = 0; j < batch_size; ++j){
					for(int k = 0; k < output_nodes; ++k){
						target_batches.Set({(int)ceil(i/batch_size), j, k, 0}, (float)target_data.matrix_values[i + (j*output_nodes) + k]);
					}
				}
		}

		std::vector<double> losses;
		for(int e = 0; e < epochs; ++e){

			for(int b = 0; b < input_batches.shape[0]; ++b){

				std::shared_ptr<Matrix> input_batch = input_batches.GetChunk({b});
				std::shared_ptr<Matrix> target_batch = target_batches.GetChunk({b});

				std::vector<std::vector<std::shared_ptr<Matrix>>> bias_deltas;
				std::vector<std::vector<std::shared_ptr<Matrix>>> weights_deltas;

				double loss;

				for(int p = 0; p < batch_size; ++p){
					std::vector<std::shared_ptr<Matrix>> sample_weights_deltas; std::vector<std::shared_ptr<Matrix>> sample_bias_deltas;

					//==================================================================//
					//																	//
					//						FEED FORWARD 								//
					//																	//
					//==================================================================//


					std::shared_ptr<Matrix> inputs = input_batch->GetChunk({p});
					std::shared_ptr<Matrix> targets = target_batch->GetChunk({p});

					std::vector<std::shared_ptr<Matrix>>& feed_forward_return_vector = feed_forward_all_template(*inputs, vectorize_inputs = false);
					std::shared_ptr<Matrix>& outputs = feed_forward_return_vector[0];
					std::vector<std::shared_ptr<Matrix>> hiddens[inner_layers.size()];
					for(int l = 1; l < inner_layers.size(); l++){
						hiddens.push_back
					}








				}




				if(print){
					std::cout << "\n" << "================" << std::endl;
					std::cout << "Epoch: " << e + 1 << std::endl;
					std::cout << "Number of Epochs: " << epochs << std::endl;
					std::cout << "Batch: " << b + 1 << std::endl;
					std::cout << "Number of Batches: " << ceil(input_array.shape[0]/batch_size) << std::endl;
					std::cout << "Batch Size: " << batch_size << std::endl;
					std::cout << "Loss: " << loss << std::endl;
					std::cout << "================" << "\n" << std::endl;
				}

			}

		}







	std::vector<std::shared_ptr<Matrix>>& feed_forward_return_vector = feed_forward_all_template()




}

























#endif
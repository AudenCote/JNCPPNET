//For helpful convnet info: 
//https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
//https://mukulrathi.com/demystifying-deep-learning/conv-net-backpropagation-maths-intuition-derivation/

#ifndef CNV_INCLUDE
#define CNV_INCLUDE


namespace CNV {

	std::shared_ptr<Matrix> convolution(Matrix& image, const int channels, const int image_width, const int image_height, Matrix& filt, Matrix& bias, const int stride, const int filter_size, const char* activation) {
		Matrix::Reshape(image, { channels, image_height, image_width });

		try {
			if (filt.dims != 4) {
				throw(std::invalid_argument("Filter must have four non-zero dimensions - number of filters, number of channels, then the filter dimensions\nException thrown in function: LRN::inter_channel()"));
			}
			else {
				for (int v : filt.shape) {
					if (v == 0) throw(std::invalid_argument("Filter must have four non-zero dimensions - number of filters, number of channels, then the filter dimensions\nException thrown in function: LRN::inter_channel()"));
				}
			}
			if (image.dims != 3) {
				throw(std::invalid_argument("Image must have three non-zero dimensions - number of channels, image width, and image height\nException thrown in function: LRN::inter_channel()"));
			}
			else {
				for (int v : filt.shape) {
					if (v == 0) throw(std::invalid_argument("Image must have three non-zero dimensions - number of channels, image width, and image height\nException thrown in function: LRN::inter_channel()"));
				}
			}
			if (bias.dims != 2) {
				throw(std::invalid_argument("Bias matrix must have two non-zero dimensions\nException thrown in function: LRN::inter_channel()"));
			}
			else {
				for (int v : bias.shape) {
					if (v == 0) throw(std::invalid_argument("Bias matrix must have two non-zero dimensions\nException thrown in function: LRN::inter_channel()"));
				}
			}

			int out_dim = (int)((image.shape[1] - filter_size) / stride) + 1;

			std::vector<int> output_matrix_shape = { filt.shape[0], out_dim, out_dim };
			std::shared_ptr<Matrix> output_matrix = std::make_shared<Matrix>(output_matrix_shape);

			for (int f = 0; f < filt.shape[0]; f++) {
				int curr_y = 0; int out_y = 0;
				while (curr_y + filt.shape[3] <= image.shape[2]) {
					int curr_x = 0; int out_x = 0;
					while (curr_x + filt.shape[2] <= image.shape[1]) {

						my_misc_utils::better_initializer_list<my_misc_utils::better_initializer_list<int>> args = my_misc_utils::make_useful({
							my_misc_utils::make_useful({ 0, image.shape[0]}),
							my_misc_utils::make_useful({ curr_y, filt.shape[2]}),
							my_misc_utils::make_useful({ curr_x, curr_x + filt.shape[3]})
							});

						std::shared_ptr<Matrix> image_section = image.GetChunk(args);

						for (int i = 0; i < filt.GetChunk({ f })->shape.size(); i++) {
							if (filt.GetChunk({ f })->shape[i] != image_section->shape[i]) {
								throw(std::length_error("Image section matrix shape does not match filter shape\nException thrown in function: LRN::inter_channel()"));
							}
						}

						std::shared_ptr<Matrix> kernel_times_image_section = Matrix::ElementwiseMultiplication(filt.GetChunk[f], *image_section);
						float summed = kernel_times_image_section->Sum();
						output_matrix->SetVal({ f, out_y, out_x }, summed + bias.GetVal({ f, 0 }));
						curr_x = curr_x + stride;
						out_x = out_x + 1;
					}
					curr_y = curr_y + stride;
					out_y = out_y + 1;
				}
			}

			if (activation == "Relu" || activation == "ReLU" || activation == "relu") {
				Matrix::ReLU(output_matrix);
			}
			else if (activation == "Sigmoid" || activation == "sigmoid") {
				Matrix::Sigmoid(output_matrix);
			}

			return output_matrix;
		}
		catch (const std::invalid_argument& e) {
			Logger::Error(e.what());
			return nullptr;
		}
		catch (const std::length_error& e) {
			Logger::Error(e.what());
			return nullptr;
		}
	}

	std::vector<std::shared_ptr<Matrix>> convolution_backprop(Matrix& dconv_prev, Matrix& conv_in, Matrix& filters, const int stride) {
		try{

			//Reshaping conv_in vector into shape expected as input to current conv layer on FOP
			//DO THIS

			//Reshaping dconv_prev vector into shape outputted by current conv layer on FOP
			//DO THIS
		
			std::shared_ptr<Matrix> dout = std::make_shared<Matrix>(conv_in.shape);
			std::shared_ptr<Matrix> dfilt = std::make_shared<Matrix>(filters.shape);
			std::vector<int> db_shape = {filters.shape[0], 1};  std::shared_ptr<Matrix> dbias = std::make_shared<Matrix>(db_shape);

			for (int f = 0; f < filters.shape[0]; f++) {
				int curr_y = 0; int out_y = 0;
				while (curr_y + filters.shape[3] <= conv_in.shape[2]) {
					int curr_x = 0; int out_x = 0;
					while (curr_x + filters.shape[2] <= conv_in.shape[1]) {

						my_misc_utils::better_initializer_list<my_misc_utils::better_initializer_list<int>> args = my_misc_utils::make_useful({
							my_misc_utils::make_useful({ 0, conv_in.shape[0]}),
							my_misc_utils::make_useful({ curr_y, curr_y + filters.shape[2]}),
							my_misc_utils::make_useful({ curr_x, curr_x + filters.shape[3]})
							});

						std::shared_ptr<Matrix> conv_in_chunk = conv_in.GetChunk(args);

						conv_in_chunk->Multiply(dconv_prev.GetVal({ f, out_y, out_x }));
						if (conv_in_chunk->num_vals != dfilt->num_vals) {
							throw std::logic_error("Mismatching matrix dimensions in funtion CNV::convolution_backprop -- Error Code 0");
						}
						for (int i = 0; i < dfilt->shape[0] * f; i++) {
							dfilt->matrix_values[i] += conv_in_chunk->matrix_values[i];
						}

						std::shared_ptr<Matrix> filter_scaled = filters.GetChunk({ f });
						filter_scaled->Multiply(dconv_prev.GetVal({ f, out_y, out_x }));

						for (int c = 0; c < dout->shape[0]; c++) {
							std::vector<float> dout_chunk_vals;
							for (int j = curr_x; j < curr_x + filters.shape[2]; j++) {
								for (int k = curr_y; k < curr_y + filters.shape[3]; k++) {
									dout_chunk_vals.push_back(dout->GetVal({ c, j, k }));
								}
							}

							if (filter_scaled->num_vals != dout_chunk_vals.size()) {
								throw std::logic_error("Mismatching matrix dimensions in funtion CNV::convolution_backprop -- Error Code 0");
							}

							for (int v = 0; v < dout_chunk_vals.size(); v++) {
								dout_chunk_vals[v] = dout_chunk_vals[v] + filter_scaled->matrix_values[v];
							}

							dout->SetChunk({ c }, dout_chunk_vals);
						}

						curr_x += stride;
						out_x += 1;
					}

					curr_y += stride;
					out_y += 1;
				}
			}

			std::vector<std::shared_ptr<Matrix>> return_vec = { dfilt, dbias, dout };  return return_vec;
		}
		catch(std::logic_error e){
			Logger::Error(e.what());
		}
	}



	std::shared_ptr<Matrix> genpool(const int type, Matrix& image, const int channels, const int image_width, const int image_height, const int filter_size = 2, const int stride = 2) {
		Matrix::Reshape(image, { channels, image_height, image_width });

		try {
			int out_dims = (int)((image.shape[1] - filter_size) / stride) + 1;

			std::vector<int> downsampled_size = { image.shape[0], out_dims, out_dims }; std::shared_ptr<Matrix> downsampled = std::make_shared<Matrix>(downsampled_size);

			for (int chan = 0; chan < image.shape[0]; chan++) {
				int curr_y = 0; int out_y = 0;
				while(curr_y + filter_size <= image.shape[1]){
					int curr_x = 0; int out_x = 0;
					while(curr_x + filter_size <= image.shape[2]){

						my_misc_utils::better_initializer_list<my_misc_utils::better_initializer_list<int>> args = my_misc_utils::make_useful({
							my_misc_utils::make_useful({ 0, image.shape[0]}),
							my_misc_utils::make_useful({ curr_y, curr_y + filter_size}),
							my_misc_utils::make_useful({ curr_x, curr_x + filter_size})
							});

						std::shared_ptr<Matrix> image_section = image.GetChunk(args);

						if (type == 0) downsampled->SetVal({chan, out_y, out_x}, Matrix::Max(*image_section)); //type zero corresponds to max-pooling
						if (type == 1) downsampled->SetVal({chan, out_y, out_x}, Matrix::Average(*image_section)); //type one corresponds to avg-pooling
						curr_x += stride;
						out_x += 1;
					}
					curr_y += stride;
					out_y += 1;
				}
			}

			return downsampled;
		}
		catch (const std::length_error& e) {
			Logger::Error(e.what());
			return nullptr;
		}
	}

	std::shared_ptr<Matrix> maxpool(Matrix& image, const int channels, const int image_width, const int image_height, const int filter_size = 2, const int stride = 2) {
		return genpool(0, image, channels, image_width, image_height, filter_size, stride);
	}

	std::shared_ptr<Matrix> avgpool(Matrix& image, const int channels, const int image_width, const int image_height, const int filter_size = 2, const int stride = 2) {
		return genpool(1, image, channels, image_width, image_height, filter_size, stride);
	}

	std::shared_ptr<Matrix> global_avgpool(Matrix& image, const int channels, const int image_width, const int image_height) {
		Matrix::Reshape(image, { channels, image_height, image_width });

		std::vector<int> ds_shape = { channels, 1, 1 }; std::shared_ptr<Matrix> downsampled = std::make_shared<Matrix>(ds_shape);

		for (int chan = 0; chan < image.shape[0]; chan++) {
			float channel_average = Matrix::Average(image.GetChunk({ chan }));

			downsampled->SetVal({ chan, 0, 0 }, channel_average);
		}

		return downsampled;
	}

	std::shared_ptr<Matrix> maxpool_backprop(Matrix& dpool, Matrix& conv_in, int pool_f, int pool_s) { //conv2 is output of most recent conv layer on the feed forward, pool_f is the filter size, pool_s is stride

		try {
			std::shared_ptr<Matrix> dout = std::make_shared<Matrix>(conv_in.shape);

			for (int c = 0; c < conv_in.shape[0]; c++) {
				int curr_y = 0; int out_y = 0;
				while (curr_y + pool_f < conv_in.shape[1]) {
					int curr_x = 0; int out_x = 0;
					while (curr_x + pool_f < conv_in.shape[2]) {

						//Always want to take the same shape from each channel (e.g. r, g, and b), but do not want to take all x values or all y values

						my_misc_utils::better_initializer_list<my_misc_utils::better_initializer_list<int>> args = my_misc_utils::make_useful({ 
							my_misc_utils::make_useful({ 0, conv_in.shape[0]}), 
							my_misc_utils::make_useful({ curr_y, curr_y + pool_f}),
							my_misc_utils::make_useful({ curr_x, curr_x + pool_f})
							});

						std::shared_ptr<Matrix> conv_in_sec = conv_in.GetChunk(args);

						std::vector<int> nanargmax_unraveled = Matrix::NanArgmax(*conv_in_sec);

						dout->SetVal({ c, curr_y + nanargmax_unraveled[0], curr_x + nanargmax_unraveled[1] }, dpool.GetVal({ c, out_y, out_x })); //This is correct because the default values when dout is created are zeroes, so filter can be bypassed

						curr_x += pool_s;
						out_x += 1;
					}
					curr_y += pool_s;
					out_y += 1;
				}
			}

			return dout;
		}
		catch (std::length_error e) {
			Logger::Error(e.what());
			return nullptr;
		}
	}

	std::shared_ptr<Matrix> avgpool_backprop(Matrix& dpool, Matrix& conv_in, const int pool_f, const int pool_s) { //conv2 is output of most recent conv layer on the feed forward, pool_f is the filter size, pool_s is stride

		//Need to calculate algorithm for expanding L1Avg to L1 size given filter size and stride. For this ONLY DPOOL IS NEEDED, NOT CONV_IN. This makes things easier in train.h

	}

	std::shared_ptr<Matrix> globavgpool_backprop(Matrix& dpool, Matrix& conv_in, const int pool_f, const int pool_s) { //conv2 is output of most recent conv layer on the feed forward, pool_f is the filter size, pool_s is stride

		std::shared_ptr<Matrix> dout = std::make_shared<Matrix>(conv_in.shape);

		for (int chan = 0; chan < dout->shape[0]; chan++) {
			float channel_average = Matrix::Average(dout->GetChunk(my_misc_utils::make_useful({ my_misc_utils::make_useful({ chan }) })));
			std::vector<int> dout_channel_shape = {dout[1], dout[2]};
			std::shared_ptr<Matrix> averaged_channel = std::make_shared<Matrix>(dout_channel_shape);
			for (int v = 0; v < averaged_channel->num_vals; v++) {
				averaged_channel->matrix_values[v] = channel_average;
			}
			dout->SetChunk({ chan }, *averaged_channel);
		}

		return dout;
	}



}


#endif
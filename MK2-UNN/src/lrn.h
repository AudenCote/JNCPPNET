//Local Response Normalization Resources:
//https://prateekvjoshi.com/2016/04/05/what-is-local-response-normalization-in-convolutional-neural-networks/
//https://towardsdatascience.com/difference-between-local-response-normalization-and-batch-normalization-272308c034ac
//https://en.wikipedia.org/wiki/Lateral_inhibition

#ifndef LRN_INCLUDE
#define LRN_INCLUDE

namespace LRN {

	std::shared_ptr<Matrix> intra_channel(Matrix& inp_matrix, const int channels, const int image_width, const int image_height, const float epsilon, const float alpha, const float beta, const float radius) {
		Matrix& input_matrix = *Matrix::Reshape(inp_matrix, { channels, image_width, image_height });

		try {
			if (input_matrix.dims != 3) {
				throw(std::invalid_argument("Invalid input matrix dimensions: input matrix must have three non-zero dimensions\nException thrown in function: LRN::intra_channel()"));
			}
			else {
				for (int v : input_matrix.shape) {
					if (v == 0) throw(std::invalid_argument("Invalid input matrix dimensions: input matrix must have three non-zero dimensions\nException thrown in function: LRN::intra_channel()"));
				}
			}
			if (radius <= 1) {
				throw(std::invalid_argument("Invalid radius (i.e. normalization neighborhood, window size) value\nException thrown in function: LRN::intra_channel()"));
			}

			std::shared_ptr<Matrix> output_matrix = std::make_shared<Matrix>(input_matrix.shape); output_matrix->matrix_values = input_matrix.matrix_values;

			for (int chan = 0; chan < input_matrix.shape[0]; chan++) {
				for (int x = 0; x < input_matrix.shape[1]; x++) {
					for (int y = 0; y < input_matrix.shape[2]; y++) {
						float normalized = input_matrix.GetVal({ chan, x, y });

						float outer_sum = 0;
						for (int i = (int)((0 > (x - radius / 2)) ? 0 : (x - radius / 2)); i < (int)((input_matrix.shape[1] < (x + radius / 2)) ? input_matrix.shape[1] : (x + radius / 2)); i++) {
							float inner_sum = 0;
							for (int j = (int)((0 > (y - radius / 2)) ? 0 : (y - radius / 2)); j < (int)((input_matrix.shape[2] < (y + radius / 2)) ? input_matrix.shape[2] : (y + radius / 2)); j++) {
								inner_sum = inner_sum + ((float)input_matrix.GetVal({ chan, i, j }) * (float)input_matrix.GetVal({ chan, i, j }));
							}
							outer_sum += inner_sum;
						}

						normalized = normalized / pow((float)(epsilon + alpha * outer_sum), float(beta));
						output_matrix->SetVal({ chan, x, y }, normalized);
					}
				}
			}

			return output_matrix;
		}
		catch (const std::invalid_argument& e) {
			Logger::Error(e.what());
			return nullptr;
		}

		
	}

	std::shared_ptr<Matrix> inter_channel(Matrix& inp_matrix, const int channels, const int image_width, const int image_height, const float epsilon, const float alpha, const float beta, const float radius) {
		Matrix& input_matrix = *Matrix::Reshape(inp_matrix, { channels, image_width, image_height });

		try {
			if (input_matrix.dims != 3) {
				throw(std::invalid_argument("Invalid input matrix dimensions: input matrix must have three non-zero dimensions\nException thrown in function: LRN::inter_channel()"));
			}
			else {
				for (int v : input_matrix.shape) {
					if (v == 0) throw(std::invalid_argument("Invalid input matrix dimensions: input matrix must have three non-zero dimensions\nException thrown in function: LRN::inter_channel()"));
				}
			}
			if (radius <= 1) {
				throw(std::invalid_argument("Invalid radius (i.e. normalization neighborhood, window size) value\nException thrown in function: LRN::inter_channel()"));
			}

			std::shared_ptr<Matrix> output_matrix = std::make_shared<Matrix>(input_matrix.shape); output_matrix->matrix_values = input_matrix.matrix_values;

			for (int chan = 0; chan < input_matrix.shape[0]; chan++) {
				for (int x = 0; x < input_matrix.shape[1]; x++) { //x is depthwise
					for (int y = 0; y < input_matrix.shape[2]; y++) {
						float normalized = input_matrix.GetVal({ chan, x, y });

						float summed = 0;
						for (int j = (int)((0 > (x - radius / 2)) ? 0 : (x - radius / 2)); j < (int)((input_matrix.shape[1] < (x + radius / 2)) ? input_matrix.shape[1] : (x + radius / 2)); j++) {
							summed = summed + pow((float)input_matrix.GetVal({ chan, j, y }), 2);
						}

						normalized = normalized / pow((float)(epsilon + alpha * summed), float(beta));
						output_matrix->SetVal({ chan, x, y }, normalized);
					}
				}
			}

			return output_matrix;
		}
		catch (const std::invalid_argument& e) {
			Logger::Error(e.what());
			return nullptr;
		}

	}
}


#endif
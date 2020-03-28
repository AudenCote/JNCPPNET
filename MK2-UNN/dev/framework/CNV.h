#ifndef CNV_INCLUDE
#define CNV_INCLUDE

#include "../lib/matrix.h"

class CNV {

	CNV() { std::cout << "Improper use of CNV class" << std::endl; }

	std::shared_ptr<Matrix> convolution(const Matrix& image, const Matrix& filt, const Matrix& bias, const int stride, const float alpha = .01f, const filter_size = 5) {

		try {
			if(filt.dims != 4){
				throw(std::invalid_argument("Filter must have four non-zero dimensions - number of filters, number of channels, then the filter dimensions\nException thrown in function: LRN::inter_channel()"));
			}else {
				for(int v : filt.shape){
					if(v == 0) throw(std::invalid_argument("Filter must have four non-zero dimensions - number of filters, number of channels, then the filter dimensions\nException thrown in function: LRN::inter_channel()"));
				}
			}
			if(image.dims != 3){
				throw(std::invalid_argument("Image must have three non-zero dimensions - number of channels, image width, and image height\nException thrown in function: LRN::inter_channel()"));
			}else {
				for(int v : filt.shape){
					if(v == 0) throw(std::invalid_argument("Image must have three non-zero dimensions - number of channels, image width, and image height\nException thrown in function: LRN::inter_channel()"));
				}
			}
			if(bias.dims != 2){
				throw(std::invalid_argument("Bias matrix must have two non-zero dimensions\nException thrown in function: LRN::inter_channel()"));
			}else {
				for(int v : bias.shape){
					if(v == 0) throw(std::invalid_argument("Bias matrix must have two non-zero dimensions\nException thrown in function: LRN::inter_channel()"));
				}
			}

			int out_dim = (int)((in_dim - f)/stride) + 1;

			std::vector<int> output_matrix_shape = {filt.shape[0], out_dim, out_dim};
			std::shared_ptr<Matrix> output_matrix = std::make_shared(output_matrix_shape);

			for(int f = 0; f < filt.shape[0]; f++){
				int curr_y = 0; int out_y = 0;
				while(curr_y + filt.shape[3] <= image.shape[2]){
					int curr_x = 0; int out_x = 0;
					while(curr_x + filt.shape[2] <= image.shape[1]){
						std::vector<int> image_section_shape = {img.shape[0], filt.shape[2], filt.shape[3]}; Matrix image_section = Matrix(image_section_shape);
						std::vector<float> ismat_vals = {};
						for(int c = 0; c < img.shape[0]; c++){
							for(int r = curr_y; r < curr_y + filt.shape[3]; r++){
								for(int v = curr_x; v < curr_x + filt.shape[2]; v++){
									ismat_vals.push_back(image.GetVal({c, v, r}));
								}
							}
						}

						if(image_section.num_vals != ismat_vals.size()){
							throw(std::length_error("Image section matrix shapes do not match properly\nException thrown in function: LRN::inter_channel()"));
						}
						for(int i = 0; i < filt.getChunk({f}).shape.size(); i++){
							if(filt.GetChunk({f}).shape[i] != image_section.shape[i]){
								throw(std::length_error("Image section matrix shape does not match filter shape\nException thrown in function: LRN::inter_channel()"));
							}
						}

						image_section.matrix_values = ismat_vals;
						std::shared_ptr<Matrix> kernel_times_image_section = Matrix::ElementwiseMultiplication(filt.GetChunk[f], image_section);
						float summed = kernel_times_image_section->Sum();
						output_matrix->Set({f, out_y, out_x}, summed + bias.GetVal({f, 0}))
						curr_x = curr_x + stride;
						out_x = out_x + 1;
					}
					curr_y = curr_y + stride;
					out_y = out_y + 1;
				}
			}

			return output_matrix;
		}
		catch(const std::invalid_argument& e) {
			Logger::Error(e.what());
			return nullptr; 
		} 
		catch(const std::length_error& e) {
			Logger::Error(e.what());
			return nullptr; 
		} 
	}

	std::shared_ptr<Matrix> maxpool(const Matrix& image, const int filter_size = 2, const int stride = 2){

		try {
			int out_dims = (int)((image[1] - filter_size)/stride) + 1;

			std::vector<int> downsampled_size = {image[0], out_dims, out_dims}; std::shared_ptr<Matrix> downsampled = std::make_shared<Matrix>(downsampled_size);

			for(int chan = 0; chan < image.shape[0]; chan++){
				int curr_y = 0; int out_y = 0;
				while curr_y + filter_size <= image.shape[1]{
					int curr_x = 0; int out_x = 0;
					while curr_x + filter_size <= image[2]{
						std::vector<int> image_section_shape = {img.shape[0], filter_size, filter_size}; Matrix image_section = Matrix(image_section_shape);
						std::vector<float> ismat_vals = {};
						for(int c = 0; c < img.shape[0]; c++){
							for(int r = curr_y; r < curr_y + filt.shape[3]; r++){
								for(int v = curr_x; v < curr_x + filt.shape[2]; v++){
									ismat_vals.push_back(image.GetVal({c, v, r}));
								}
							}
						}

						if(image_section.num_vals != ismat_vals.size()){
							throw(std::length_error("Image section matrix shapes do not match properly"));
						}
						for(int i = 0; i < filt.getChunk({f}).shape.size(); i++){
							if(filt.GetChunk({f}).shape[i] != image_section.shape[i]){
								throw(std::length_error("Image section matrix shape does not match filter shape"));
							}
						}

						image_section.matrix_values = ismat_vals;
						downsampled.Set({chan, out_y, out_x}, Matrix::Max(image_section));
						curr_x = curr_x + stride;
							out_x = out_x + 1;
					}
					curr_y = curr_y + stride;
					out_y = out_y + 1;
				}
			}
		}
		catch(const std::length_error& e){
			Logger::Error(e.what());
		}


	}

}



#endif
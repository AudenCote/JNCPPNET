//Local Response Normalization Resources:
//https://prateekvjoshi.com/2016/04/05/what-is-local-response-normalization-in-convolutional-neural-networks/
//https://towardsdatascience.com/difference-between-local-response-normalization-and-batch-normalization-272308c034ac
//https://en.wikipedia.org/wiki/Lateral_inhibition


#include "../lib/matrix.h"
#include <cmath>

class LRN{
public:
	LRN() { std::cout << "Improper usage of LRN class" << std::endl; }

	static Matrix intra_channel(const Matrix& input_matrix, const float epsilon, const float alpha, const float beta, const float radius){

		try{
			if(input_matrix.dims != 3) {
				throw(std::invalid_argument("Invalid input matrix dimensions: input matrix must have three non-zero dimensions"));
			}else{
				for(int v : input_matrix.shape){
					if(v == 0) throw(std::invalid_argument("Invalid input matrix dimensions: input matrix must have three non-zero dimensions"));
				}
			}
			if(radius <= 1){
				throw(std::invalid_argument("Invalid radius (i.e. normalization neighborhood, window size) value"))
			}

			Matrix output_matrix = input_matrix;

			for(int chan = 0; chan < input_matrix.shape[0]; chan++){
				for(int x = 0; x < input_matrix.shape[1]; x++){
					for(int y = 0; y < input_matrix.shape[2]; y++){
						float normalized = input_matrix.GetVal({chan, x, y});

						float outer_sum = 0;
						for(i = std::max(0, x - radius/2); i < std::min(input_matrix.shape[1], x + radius/2); i++){
							float inner_sum = 0;
							for(j = std::max(0, y - radius/2); j < std::min(input_matrix.shape[2], y + radius/2); j++){
								inner_sum = inner_sum + pow((float)input_matrix.GetVal({chan, i, j}, 2));
							}
							outer_sum += inner_sum;
						}
						
						normalized = normalized/pow((float)(epsilon + alpha*outer_sum), float(beta));
						output_matrix.Set({chan, x, y}, normalized);
					}
				}
			}

		}
		catch(const std::invalid_argument& e) {
			std::cout << e.what() << "\nException thrown in function: LRN::intra_channel()" << std::endl;
			return nullptr; 
		} 

		return output_matrix;
	}

	static Matrix inter_channel(Matrix& input_matrix, float epsilon, float alpha, float beta, float radius){

		try{
			if(input_matrix.dims != 3) {
				throw(std::invalid_argument("Invalid input matrix dimensions: input matrix must have three non-zero dimensions"));
			}else{
				for(int v : input_matrix.shape){
					if(v == 0) throw(std::invalid_argument("Invalid input matrix dimensions: input matrix must have three non-zero dimensions"));
				}
			}
			if(radius <= 1){
				throw(std::invalid_argument("Invalid radius (i.e. normalization neighborhood, window size) value"))
			}

			Matrix output_matrix = input_matrix;

			for(int chan = 0; chan < input_matrix.shape[0]; chan++){
				for(int x = 0; x < input_matrix.shape[1]; x++){ //x is depthwise
					for(int y = 0; y < input_matrix.shape[2]; y++){
						float normalized = input_matrix.GetVal({chan, x, y});

						float summed = 0;
						for(int j = std::max(0, x - radius/2); j < std::min(input_matrix[1] - 1, x + radius/2); j++){
							summed = summed + pow((float)input_matrix.GetVal({chan, j, y}), 2);
						}
						
						normalized = normalized/pow((float)(epsilon + alpha*summed), float(beta));
						output_matrix.Set({chan, x, y}, normalized);
					}
				}
			}
			

		}
		catch(const std::invalid_argument& e) {
			std::cout << e.what() << "\nException thrown in function: LRN::inter_channel()" << std::endl;
			return nullptr; 
		} 

		return output_matrix;

	}
}
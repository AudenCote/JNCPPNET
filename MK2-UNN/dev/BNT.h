//Batch normalizing transform resources:
//https://towardsdatascience.com/difference-between-local-response-normalization-and-batch-normalization-272308c034ac
//https://gist.github.com/shagunsodhani/4441216a298df0fe6ab0
//https://stats.stackexchange.com/questions/174295/what-does-the-term-saturating-nonlinearities-mean
//https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c
//https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html

#include "../lib/matrix.h"
#include <cmath>

class BNT{
public:

	BNT() { std::cout << "Improper usage of LRN class" << std::endl; }

	static shared_ptr<Matrix> normalize_batch(const Matrix& input_batch, const float gamma, const float beta, const double epsilon = .000000001) {
		try {
			if(input_batch.dims != 4 && input_batch.dims != 3){
				throw(std::invalid_argument("Invalid matrix dimensions: input matrix must have either three or four non-zero dimensions (a full batch of samples of arbitrary length)"));
			}else{
				for(int v : input_batch.shape){
					if(v == 0) throw(std::invalid_argument("Invalid input matrix dimensions: input matrix must have either three or four non-zero dimensions (a full batch of samples of arbitrary length)"));
				}
			}

			Matrix avgd_batch = Matrix(input_batch.shape);
			Matrix denom_batch = Matrix(input_batch.shape);
			if(input_batch.dims == 4){
				for(int c = 0; c < input_batch.shape[1]; c++){
					for(int x = 0; x < input_batch.shape[2]; x++){
						for(int y = 0; y < input_batch.shape[3]){
							for(int s = 0; s < input_batch.shape[0]; s++){
								float sum = 0;
								for(int s = 0; s < input_batch.shape[0]; s++){
									sum = sum + input_batch.GetVal({s, c, x, y});
								}
								float avg = sum/input_batch.shape[0];

								float stdev_sum = 0;
								for(int s = 0; s < input_batch.shape[0]; s++){
									stdev_sum = stdev_sum + pow((float)(input_batch.GetVal({s, c, x, y}) - avg), 2);
								}
								float stdev = stdev_sum/input_batch.shape;
								float denom = pow(stdev, .5) + epsilon;

								avgd_batch.Set({s, c, x, y}, avg);
								denom_batch.Set({s, c, x, y}, denom);
							}
						}
					}
				}
			}else if(input_batch.dims == 3){
				for(int x = 0; x < input_batch.shape[2]; x++){
					for(int y = 0; y < input_batch.shape[3]; y++){
						float sum = 0;
						for(int s = 0; s < input_batch.shape[0]; s++){
							sum = sum + input_batch.GetVal({s, x, y});
						}
						float avg = sum/input_batch.shape[0];

						float stdev_sum = 0;
						for(int s = 0; s < input_batch.shape[0]; s++){
							stdev_sum = stdev_sum + pow((float)(input_batch.GetVal({s, x, y}) - avg), 2);
						}
						float stdev = stdev_sum/input_batch.shape;
						float denom = pow(stdev, .5) + epsilon;

						avgd_batch.Set({s, x, y}, avg);
						denom_batch.Set({s, x, y}, denom);
					}
				}
			}
			shared_ptr<Matrix> output_batch = Matrix::ElementwiseSubtraction(input_batch, avgd_batch);
			output_batch = Matrix::ElementwiseDivision(*output_batch, denom_batch);
			output_batch.Multiply(gamma);
			output_batch.Add(beta);
		}
		catch(const std::invalid_argument& e) {
			std::cout << e.what() << "\nException thrown in function: LRN::intra_channel()" << std::endl;
			return nullptr; 
		} 
		
		return output_batch;
	}
}
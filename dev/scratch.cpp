#include "../lib/matrix.h"

int main(){

	std::vector<float> m1_vals = {-0.593824}; std::vector<int> m1_shape = {1, 1};
	std::vector<float> m2_vals = {-0.299417, 0.791925, 0.64568}; std::vector<int> m2_shape = {3, 1};

	Matrix m1 = Matrix(m1_shape); 
	m1.matrix_values = m1_vals;
	Matrix m2 = Matrix(m2_shape); 
	m2.matrix_values = m2_vals;

	std::shared_ptr<Matrix> m3 = Matrix::DotProduct(m2, m1);

	// for(float v : m3->matrix_values){
	// 	std::cout << v << std::endl;
	// }

	return 0;

}


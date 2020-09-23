#include <vector>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <string>
#include <memory>
#include <matrix.h>
#include <structure.h>
#include <handle_trainables.h>
#include <methods.h>
#include <fcl.h>
#include <cnv.h>
#include <lrn.h> 
#include <bnt.h>
#include <predict.h>
#include <train.h>


int main() {

	std::vector<int> input_matrix_shape = {8, 1};
	Matrix input_matrix = Matrix(input_matrix_shape);

	NeuralNetwork NN = NeuralNetwork(8, 4, .1);

	NN.InputLayer(8);
	NN.FullyConnected(16, "sigmoid");
	NN.OutputLayer(4, "sigmoid");

	NN.Predict(input_matrix, false);




	return 0;
}

#include "DNN.h"

int main() {

    NeuralNetwork NN = NeuralNetwork(2, 2, .1f);
    NN.HiddenLayer(3);
    NN.InitializeParameters();

    std::vector<int> data_shape = {3, 1}; Matrix data = Matrix(data_shape);

    NN.Predict(data);

    NN.Deallocate();


    std::cout << "Returning" << std::endl;
    return 0;
}
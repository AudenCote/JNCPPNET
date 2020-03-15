#include "DNN.h"

int main() {

    NeuralNetwork NN = NeuralNetwork(2, 2, .1f);
    NN.HiddenLayer(3);
    NN.InitializeParameters();

    std::vector<float> in_vals = {1, 0};
    std::vector<float> tar_vals = {1, 0};

    std::vector<int> data_shape = {2, 1, 1}; Matrix data = Matrix(data_shape);
    std::vector<int> tar_shape = {2, 1, 1}; Matrix tar = Matrix(data_shape);

    data.matrix_values = in_vals;
    tar.matrix_values = tar_vals;

    NN.feed_and_propogate(data, tar, 5, 1);

    std::cout << NN.Predict(*data.GetChunk({1}))->GetVal({0, 0}) << std::endl;

    return 0;
}
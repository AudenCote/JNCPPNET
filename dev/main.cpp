#include "DNN.h"

int main() {

    NeuralNetwork NN = NeuralNetwork(2, 2, .1f);
    NN.HiddenLayer(3);
    NN.InitializeParameters();

    std::vector<int> data_shape = {16, 2, 1}; Matrix data = Matrix(data_shape); data.Randomize();
    std::vector<int> tar_shape = {16, 2, 1}; Matrix tar = Matrix(data_shape); tar.Randomize();

    NN.feed_and_propogate(data, tar, 1, 8);

    NN.Predict(data);

    return 0;
}
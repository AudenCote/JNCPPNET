#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <memory>
#include <ctime>
#include <cmath>
#include <utils.h>
#include <matrix.h>
#include <DNN.h>

int main() {

    NeuralNetwork NN = NeuralNetwork(2, 1, .1f);
    NN.HiddenLayer(8);
    NN.InitializeParameters();

    std::vector<float> in_vals = { 1, 0, 1, 1, 0, 1, 0, 0 };
    std::vector<float> tar_vals = { 0, 0, 0, 0 };

    std::vector<int> data_shape = { 4, 2, 1 }; Matrix data = Matrix(data_shape);
    std::vector<int> tar_shape = { 4, 1, 1 }; Matrix tar = Matrix(data_shape);

    data.matrix_values = in_vals;
    tar.matrix_values = tar_vals;

    NN.feed_and_propogate(data, tar, 100, 1);

    std::cout << "\nPrediction: " << NN.Predict(*data.GetChunk({ 0 }))->GetVal({ 0, 0 }) << std::endl;

    return 0;
}

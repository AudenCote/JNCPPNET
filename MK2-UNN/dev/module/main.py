from cython_file import sum_my_vector_cpp

NN = NeuralNetwork(input_nodes = 2, output_nodes = 2, learning_rate = .1)
NN.Convolutional(image_width = 28, image_height = 28, filter_size = 5, filters = 8, stride = 2, channels = 3)
NN.MaxPooling(filter_size = 5, stride = 2)
NN.LocalResponseNormalization(type = "inter-channel", epsilon = .001, alpha = 1, beta = 1, radius = 2)
NN.FullyConnected(nodes = 64)
NN.Initialize()

NN.Predict(input_vector = [], target_vector = [])
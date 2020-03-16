import random
from matrix import *
from DNN import *

test_in = [[1, 0],
		   [1, 1],
		   [0, 1],
		   [0, 0]]
test_out = [[1], [0], [1], [0]]

NN = NeuralNetwork(2, 1, .1)
NN.hidden_layer(8)
NN.initialize_weights()
NN.train(test_in, test_out, gd_type='stochastic', iterations=1000)
NN.save_model()
prediction = NN.predict(test_in[0], NN.load_model())
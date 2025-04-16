import math
import random

class NeuralNetwork():

    def __init__(self):

        # Seed the random number generator, so we get the same random numbers each time.
        random.seed(1)
        
        # Create 3 weights and set them to random values in the range -1 to +1
        self.weights = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

    # Make a prediction
    def think(self, neuron_inputs):
        sum_of_weighted_inputs = self.__sum_of_weighted_inputs(neuron_inputs)
        neuron_output = self.__sigmoid(sum_of_weighted_inputs)
        return neuron_output

    # Adjust the weights of the neural network to minimise the error for the training set
    def train(self):
        pass

    # Calculate the sigmoid (our activation function)
    def __sigmoid(self):
        pass

    # Calculate the gradient of the sigmoid using its own output
    def __sigmoid_gradient(self):
        pass

    # Multiply each input by its own weight, and then sum the total
    def __sum_of_weighted_inputs(self):
        pass

# The neural network will use this training set of 4 examples, to learn the patern
training_set_examples = [{"inputs":[0, 0, 1], "output":0}, {"inputs":[1, 1, 1], "output":1}, {"inputs":[1, 0, 1], "output":1}, {"inputs":[0, 1, 1], "output":0}]
import numpy as np


class Model:
    def __init__(self, **kwargs):
        self.weight_matrix = list()
        self.output_matrix = list()
        self.delta_matrix = list()
        self.previous_input_shape = kwargs['input_shape']
        self.learning_rate = kwargs['learning_rate']
        print("Model is instantiated with feature vector being 1 X ", self.previous_input_shape)

    # whenever a hidden layer is added, this means that a weight matrix is initialised which is of input_shape*number_of_neurons
    def add(self, **kwargs):
        number_of_neurons = kwargs['number_of_neurons']
        weight_matrix = 2 * np.random.random((number_of_neurons, self.previous_input_shape)) - 1
        print(" hidden layer added with Dimension = ", self.previous_input_shape, " X ", number_of_neurons)
        self.weight_matrix.append(weight_matrix)
        self.previous_input_shape = number_of_neurons

    def activation(self, matrix):
        return 1.0 / (1.0 + np.exp(-matrix))

    def feedforward(self, row):
        self.output_matrix = list()
        self.delta_matrix = list()
        previous_matrix = row  # during backprop, this first row will act as input
        self.output_matrix.append(row)
        for i in self.weight_matrix:
            previous_matrix = np.matmul(previous_matrix, np.transpose(i))
            previous_matrix = self.activation(previous_matrix)
            self.output_matrix.append(previous_matrix)

    def backpropagation(self, target):
        # since this is a binary classification with final layer being a single neuron, we will pass target as a scalar.
        # Later on, we will make multi-class classification and wll pass target as vector along with error

        layers = self.weight_matrix
        output_layer = self.output_matrix[-1]
        error = output_layer[0] - target
        a = list()
        a.append(error)
        self.delta_matrix.append(a)

        for layer in reversed(range(len(layers) - 1)):
            # calculate delta of next layer with respect to this current layer
            self.calculate_delta(layer)
            # then update the next layer weights
            self.update_weight_matrix(layer + 1, self.learning_rate)
        self.update_weight_matrix(0, self.learning_rate)

    def calculate_delta(self, layer_index):
        current_neurons = self.weight_matrix[layer_index].shape[0]
        delta_current_layer = list()
        delta_next_layer = self.delta_matrix[-1]
        weight_matrix_next_layer = self.weight_matrix[layer_index + 1]
        output_matrix = self.output_matrix[layer_index + 2]

        for neuron in range(current_neurons):
            error_for_current_neuron = 0
            for n in range(len(delta_next_layer)):
                error_for_current_neuron += (delta_next_layer[n] * weight_matrix_next_layer[n][neuron] * self.pure_derivative(output_matrix[n]))
            delta_current_layer.append(error_for_current_neuron)
        self.delta_matrix.append(delta_current_layer)

    def update_weight_matrix(self, layer_index, alpha):
        weight_matrix = self.weight_matrix[layer_index]
        output_matrix = self.output_matrix[layer_index]
        current_neurons = weight_matrix.shape[0]
        previous_connections = weight_matrix.shape[1]
        for neuron in range(current_neurons):
            local_gradient = (self.output_matrix[layer_index + 1][neuron]) * (1 - self.output_matrix[layer_index + 1][neuron])
            for prev in range(previous_connections):
                weight_new = weight_matrix[neuron][prev] - alpha * self.delta_matrix[-1][neuron] * local_gradient * output_matrix[prev]
                self.weight_matrix[layer_index][neuron][prev] = weight_new

    def pure_derivative(self, x):
        return x * (1.0 - x)

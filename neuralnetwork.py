import math
import random

class NeuralNetwork():

    def __init__(self, input_neurons, hidden_neurons_1, hidden_neurons_2, output_neurons):
        # Inicializar pesos para las capas
        random.seed(1)
        self.hidden_weights_1 = [[random.uniform(-1, 1) for _ in range(input_neurons)] for _ in range(hidden_neurons_1)]
        self.hidden_weights_2 = [[random.uniform(-1, 1) for _ in range(hidden_neurons_1)] for _ in range(hidden_neurons_2)]
        self.output_weights = [random.uniform(-1, 1) for _ in range(hidden_neurons_2)]

    # Función para realizar una predicción
    def think(self, inputs):
        # Calcular las salidas de la primera capa oculta
        hidden_layer_1_outputs = [self.__sigmoid(self.__sum_of_weighted_inputs(inputs, self.hidden_weights_1[i]))
                                  for i in range(len(self.hidden_weights_1))]
        # Calcular las salidas de la segunda capa oculta
        hidden_layer_2_outputs = [self.__sigmoid(self.__sum_of_weighted_inputs(hidden_layer_1_outputs, self.hidden_weights_2[i]))
                                  for i in range(len(self.hidden_weights_2))]
        # Calcular la salida final
        output = self.__sigmoid(self.__sum_of_weighted_inputs(hidden_layer_2_outputs, self.output_weights))
        return output

    # Entrenar la red neuronal
    def train(self, training_set_examples, number_of_iterations):
        for iteration in range(number_of_iterations):
            for example in training_set_examples:
                inputs = example["inputs"]
                expected_output = example["output"]

                # Forward pass
                hidden_layer_1_outputs = [self.__sigmoid(self.__sum_of_weighted_inputs(inputs, self.hidden_weights_1[i]))
                                          for i in range(len(self.hidden_weights_1))]
                hidden_layer_2_outputs = [self.__sigmoid(self.__sum_of_weighted_inputs(hidden_layer_1_outputs, self.hidden_weights_2[i]))
                                          for i in range(len(self.hidden_weights_2))]
                output = self.__sigmoid(self.__sum_of_weighted_inputs(hidden_layer_2_outputs, self.output_weights))

                # Calcular errores
                output_error = expected_output - output
                hidden_errors_2 = [output_error * self.output_weights[i] * self.__sigmoid_gradient(hidden_layer_2_outputs[i])
                                   for i in range(len(self.hidden_weights_2))]
                hidden_errors_1 = [sum(hidden_errors_2[j] * self.hidden_weights_2[j][i] for j in range(len(self.hidden_weights_2))) *
                                   self.__sigmoid_gradient(hidden_layer_1_outputs[i]) for i in range(len(self.hidden_weights_1))]

                # Backpropagation: ajustar los pesos de la capa de salida
                for i in range(len(self.output_weights)):
                    self.output_weights[i] += hidden_layer_2_outputs[i] * output_error * self.__sigmoid_gradient(output)

                # Backpropagation: ajustar los pesos de la segunda capa oculta
                for i in range(len(self.hidden_weights_2)):
                    for j in range(len(self.hidden_weights_2[i])):
                        self.hidden_weights_2[i][j] += hidden_layer_1_outputs[j] * hidden_errors_2[i]

                # Backpropagation: ajustar los pesos de la primera capa oculta
                for i in range(len(self.hidden_weights_1)):
                    for j in range(len(self.hidden_weights_1[i])):
                        self.hidden_weights_1[i][j] += inputs[j] * hidden_errors_1[i]

    # Calcular la suma ponderada de entradas
    def __sum_of_weighted_inputs(self, inputs, weights):
        return sum(input_value * weight for input_value, weight in zip(inputs, weights))

    # Función sigmoide
    def __sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # Gradiente de la función sigmoide
    def __sigmoid_gradient(self, output):
        return output * (1 - output)


# Crear una red neuronal con 10 entradas, 6 neuronas en la primera capa oculta, 4 en la segunda capa oculta y 1 salida
neural_network = NeuralNetwork(input_neurons=10, hidden_neurons_1=6, hidden_neurons_2=4, output_neurons=1)

print("Pesos iniciales de la primera capa oculta: " + str(neural_network.hidden_weights_1))
print("Pesos iniciales de la segunda capa oculta: " + str(neural_network.hidden_weights_2))
print("Pesos iniciales de la capa de salida: " + str(neural_network.output_weights))

# Conjunto de entrenamiento más complejo
training_set_examples = [
    {"inputs": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], "output": 1},
    {"inputs": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0], "output": 0},
    {"inputs": [1, 1, 1, 1, 0, 0, 0, 0, 1, 1], "output": 1},
    {"inputs": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0], "output": 0}
]

# Entrenar la red neuronal
neural_network.train(training_set_examples, number_of_iterations=50000)

print("Pesos ajustados de la primera capa oculta: " + str(neural_network.hidden_weights_1))
print("Pesos ajustados de la segunda capa oculta: " + str(neural_network.hidden_weights_2))
print("Pesos ajustados de la capa de salida: " + str(neural_network.output_weights))

# Solicitar al usuario que ingrese los valores de la nueva situación
new_situation = input("Ingrese los valores de la nueva situación separados por comas (por ejemplo, 1,0,1,0,1,0,1,0,1,0): ")
new_situation = [int(value.strip()) for value in new_situation.split(",")]

# Realizar la predicción
prediction = neural_network.think(new_situation)
print("Predicción para la nueva situación --> " + str(prediction))
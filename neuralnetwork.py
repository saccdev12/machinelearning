import random
import math

# funciones auxiliares

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

# Red Neuronal 

class NeuralNetwork:
    def __init__(self):
        self.w_input_hidden = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(4)]
        self.w_hidden_output = [random.uniform(-1, 1) for _ in range(4)]
        self.learning_rate = 0.5

    def forward(self, inputs):
        self.hidden = []
        for h in range(4):
            suma = sum(inputs[i] * self.w_input_hidden[h][i] for i in range(4))
            self.hidden.append(sigmoid(suma))

        suma = sum(self.hidden[i] * self.w_hidden_output[i] for i in range(4))
        self.output = sigmoid(suma)

        return self.output

    def train(self, inputs, target):
        output = self.forward(inputs)
        error = target - output
        delta_output = error * dsigmoid(output)

        for i in range(4):
            self.w_hidden_output[i] += self.learning_rate * delta_output * self.hidden[i]

        deltas_hidden = []
        for h in range(4):
            err = delta_output * self.w_hidden_output[h]
            deltas_hidden.append(err * dsigmoid(self.hidden[h]))

        for h in range(4):
            for i in range(4):
                self.w_input_hidden[h][i] += self.learning_rate * deltas_hidden[h] * inputs[i]

        return error


# Datos de entrenamiento
# salida = primer bit de la entrada

patrones = [
    ([1,0,1,0], 1),
    ([0,1,0,1], 0),
    ([1,1,0,0], 1),
    ([0,0,1,1], 0),
    ([1,0,0,1], 1),
    ([0,1,1,0], 0),
]

# Entrenamiento

nn = NeuralNetwork()

print("Entrenando red neuronal...")
for epoch in range(5000):
    total_error = 0
    for entrada, salida in patrones:
        total_error += abs(nn.train(entrada, salida))
    if epoch % 1000 == 0:
        print(f"Época {epoch} | Error total: {total_error:.4f}")

# Pruebas automáticas

print("\nPruebas de la red entrenada:")
for entrada, salida in patrones:
    pred = nn.forward(entrada)
    print(f"Entrada: {entrada} | Esperado: {salida} | Predicho: {round(pred,3)}")


# INTERACCIÓN CON EL USUARIO

while True:
    print("\nIngresa 4 números binarios separados por espacio (o 'salir'):")
    entrada = input("> ")

    if entrada.lower() == "salir":
        break

    try:
        valores = list(map(int, entrada.split()))
        if len(valores) != 4 or any(v not in (0,1) for v in valores):
            print("Debes ingresar exactamente 4 números binarios (0 o 1).")
            continue
    except:
        print("Formato no válido.")
        continue

    pred = nn.forward(valores)
    print(f"Predicción de la red: {round(pred,3)}  (≈ {int(pred>=0.5)})")





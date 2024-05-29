import numpy as np


# Функция активации (пороговая функция)
def activation_function(x):
    return 1 if x >= 0 else 0


# Персептрон
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size + 1) * 0.01  # Инициализация весов
        self.learning_rate = learning_rate

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Добавление единичного смещения
        weighted_sum = np.dot(self.weights, x)
        return activation_function(weighted_sum)

    def train(self, training_data, labels, epochs=1000):
        for epoch in range(epochs):
            for x, label in zip(training_data, labels):
                prediction = self.predict(x)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * np.array(x)
                self.weights[0] += self.learning_rate * error


# Подготовка данных
# Входные данные для символов (A, B, C, D) в виде бинарных векторов (упрощенно)
training_data = [
    [0, 1, 1, 0, 1, 1, 1, 0],  # A (образ 1)
    [0, 1, 1, 0, 1, 1, 0, 0],  # A (образ 2)
    [0, 1, 1, 1, 1, 1, 1, 0],  # A (образ 3)
    [0, 1, 1, 0, 1, 0, 1, 0],  # A (образ 4)
    [1, 1, 1, 0, 1, 0, 1, 0],  # B (образ 1)
    [1, 1, 0, 0, 1, 0, 1, 0],  # B (образ 2)
    [1, 1, 1, 0, 1, 1, 1, 0],  # B (образ 3)
    [1, 1, 1, 1, 1, 0, 1, 0],  # B (образ 4)
    [1, 0, 0, 0, 1, 1, 1, 1],  # C (образ 1)
    [1, 0, 0, 0, 1, 0, 1, 1],  # C (образ 2)
    [1, 0, 0, 0, 1, 1, 1, 0],  # C (образ 3)
    [1, 0, 1, 0, 1, 1, 1, 1],  # C (образ 4)
    [1, 1, 0, 0, 1, 1, 1, 0],  # D (образ 1)
    [1, 1, 0, 1, 1, 1, 1, 0],  # D (образ 2)
    [1, 1, 0, 0, 1, 0, 1, 0],  # D (образ 3)
    [1, 1, 0, 0, 1, 1, 1, 1],  # D (образ 4)
]

# Метки для каждого символа (A, B, C, D) в виде одного горячего кодирования
labels = [
    [1, 0, 0, 0],  # A
    [1, 0, 0, 0],  # A
    [1, 0, 0, 0],  # A
    [1, 0, 0, 0],  # A
    [0, 1, 0, 0],  # B
    [0, 1, 0, 0],  # B
    [0, 1, 0, 0],  # B
    [0, 1, 0, 0],  # B
    [0, 0, 1, 0],  # C
    [0, 0, 1, 0],  # C
    [0, 0, 1, 0],  # C
    [0, 0, 1, 0],  # C
    [0, 0, 0, 1],  # D
    [0, 0, 0, 1],  # D
    [0, 0, 0, 1],  # D
    [0, 0, 0, 1],  # D
]

# Инициализация и обучение персептрона
input_size = len(training_data[0])
perceptrons = [Perceptron(input_size) for _ in range(4)]

# Обучение
for i, p in enumerate(perceptrons):
    p.train(training_data, [label[i] for label in labels])

# Тестирование
test_data = [
    [0, 1, 0, 0, 1, 1, 1, 0],  # Тестовые данные для A (новый шрифт)
    [1, 1, 1, 0, 0, 0, 1, 0],  # Тестовые данные для B (новый шрифт)
    [1, 0, 0, 0, 1, 1, 1, 0],  # Тестовые данные для C (новый шрифт)
    [1, 1, 0, 0, 1, 1, 0, 1],  # Тестовые данные для D (новый шрифт)
]

correct_labels = [
    [1, 0, 0, 0],  # A
    [0, 1, 0, 0],  # B
    [0, 0, 1, 0],  # C
    [0, 0, 0, 1],  # D
]

correct_predictions = 0
for x, correct_label in zip(test_data, correct_labels):
    outputs = [p.predict(x) for p in perceptrons]
    print(f"Input: {x} -> Predicted Output: {outputs} -> Correct Label: {correct_label}")
    if outputs == correct_label:
        correct_predictions += 1

accuracy = correct_predictions / len(test_data)
print(f"Accuracy: {accuracy * 100}%")

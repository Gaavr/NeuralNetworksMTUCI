import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Функция активации (сигмоидальная функция)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная сигмоидальной функции
def sigmoid_derivative(x):
    return x * (1 - x)

# Загрузка данных MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
X, y = mnist["data"], mnist["target"]

# Нормализация данных
X = X / 255.0
y = LabelBinarizer().fit_transform(y)  # One-hot encoding

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Параметры нейронной сети
input_size = X_train.shape[1]
output_size = y_train.shape[1]
learning_rate = 0.1
epochs = 50

# Инициализация весов и смещений
weights = np.random.uniform(-0.003, 0.003, (input_size, output_size))
bias = np.random.uniform(-0.003, 0.003, output_size)

# Обучение однослойного персептрона
for epoch in range(epochs):
    total_error = 0
    for i in range(len(X_train)):
        x = X_train[i]
        target = y_train[i]

        # Прямой проход: вычисление взвешенной суммы и выходного сигнала
        net_input = np.dot(x, weights) + bias
        output = sigmoid(net_input)

        # Вычисление ошибки
        error = target - output
        total_error += np.sum(error**2)

        # Коррекция весов и смещений
        weights += learning_rate * np.outer(x, error * sigmoid_derivative(output))
        bias += learning_rate * error * sigmoid_derivative(output)

    # Вывод информации о процессе обучения
    if (epoch + 1) % 10 == 0:
        avg_error = total_error / len(X_train)
        print(f"Epoch {epoch + 1}, Average Error: {avg_error}")

# Тестирование нейронной сети
def test_perceptron(X_test, y_test, weights, bias):
    correct_predictions = 0
    for i in range(len(X_test)):
        x = X_test[i]
        net_input = np.dot(x, weights) + bias
        output = sigmoid(net_input)
        predicted_label = np.argmax(output)
        actual_label = np.argmax(y_test[i])
        if predicted_label == actual_label:
            correct_predictions += 1
    accuracy = correct_predictions / len(X_test)
    return accuracy

# Проверка точности на тестовой выборке
accuracy = test_perceptron(X_test, y_test, weights, bias)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Вывод результатов
print("Обучение завершено.")
print(f"Точность распознавания на тестовой выборке: {accuracy * 100:.2f}%")

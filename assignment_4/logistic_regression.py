import math

import numpy as np


def sigmoid_function(z):
    """
    Evaluate sigmoid function at point z
    """
    try:
        value = 1.0 / (1 + math.exp(-1 * z))
    except:
        if(z > 0):
            return 0.9999
        else:
            return 0.0001

    return value


class LogisticRegression():
    def __init__(self, learning_rate=0.005, threshold=0.005):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.sigmoid_vectorizer = np.vectorize(sigmoid_function)

    def fit(self, x_train, y_train):
        self.input_dimension = x_train.shape[1]
        self.number_of_training_data = x_train.shape[0]

        y = y_train.reshape(self.number_of_training_data, -1)

        # Bias terms to add
        bias = np.ones((self.number_of_training_data, 1))

        # Augmented training data
        x = np.hstack((bias, x_train))

        # Randomly initialize thetas
        self.thetas = np.random.uniform(-0.5,
                                        0.5, (1, self.input_dimension + 1))

        while(True):
            z = np.dot(x, self.thetas.T)

            predicted_values = self.sigmoid_vectorizer(z)

            correction = y - predicted_values

            # Perform batch update on the data
            self.thetas = self.thetas + \
                self.learning_rate * \
                (1 / self.number_of_training_data) * np.dot(correction.T, x)

            if(np.max(correction) <= self.threshold):
                break

    def predict(self, x_predict):
        data_size = x_predict.shape[0]
        bias = np.ones((data_size, 1))
        x = np.hstack((bias, x_predict))

        z = np.dot(x, self.thetas.T)
        values = self.sigmoid_vectorizer(z)

        predicted_labels = np.array(
            [1 if value > 0.5 else 0 for value in values])

        return predicted_labels

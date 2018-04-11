import numpy as np


class LogisticRegression():
    def __init__(self, learning_rate=0.5, threshold=0.005):
        self.learning_rate = learning_rate
        self.threshold = threshold

    def fit(self, x_train, y_train):
        self.input_dimension = x_train.shape[1]
        self.number_of_training_data = x_train.shape[0]

        y = y_train.reshape(self.number_of_training_data, -1)

        # Bias terms to add
        bias = np.ones((self.number_of_training_data, 1))

        # Augmented training data
        x = np.hstack((bias, x_train))

        # Randomly initialize thetas
        self.thetas = np.random.rand(1, self.input_dimension + 1) - 0.5

        while(True):
            z = np.dot(x, self.thetas.T)

            predicted_values = 1.0 / (1 + np.exp(-1 * z))

            correction = y - predicted_values

            # Perform batch update on the data
            self.thetas = self.thetas + \
                self.learning_rate * \
                (1 / self.number_of_training_data) * np.dot(correction.T, x)

            if(np.max(correction) <= self.threshold):
                break

    def predict(self, x_predict):
        x = np.hstack((np.array([1]), x_predict))
        z = np.dot(x, self.thetas.T)
        value = 1.0 / (1 + np.exp(-1 * z))

        if(value > 0.5):
            return 1
        else:
            return 0

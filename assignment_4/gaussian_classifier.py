import math

import numpy as np


def gaussian_distribution_value(x, mu, sigma_inv, sigma_det):
    """
    Evaluate gaussian function at point x
    """

    x_minus_mu = x - mu

    exponent = np.exp(
        -np.dot(
            np.dot(x_minus_mu.T, sigma_inv),
            x_minus_mu
        ) / 2.0
    )

    value = exponent / (math.sqrt(pow(2 * math.pi, len(x)) * sigma_det))

    return value


class GaussianClassifier():

    def fit(self, x_train, y_train):
        # Store the parameters to use with each class
        self.parameters = []

        self.number_of_training_data = len(x_train)
        self.labels = set(list(y_train))

        for label in self.labels:
            class_indices = np.where(y_train == label)
            class_prior = len(class_indices[0]) / self.number_of_training_data

            class_data = x_train[class_indices]

            class_mean = np.mean(class_data, axis=0)

            covariance = np.cov(class_data, rowvar=False)

            covariance_inverse = np.linalg.inv(covariance)
            covariance_det = np.linalg.det(covariance)

            class_parameter_dictionary = {
                "prior": class_prior,
                "mean": class_mean,
                "covariance_inverse": covariance_inverse,
                "covariance_det": covariance_det,
                "label": label
            }

            self.parameters.append(class_parameter_dictionary)

    def predict(self, x_predict):
        probabilites = []

        for parameter in self.parameters:
            class_likelihood = gaussian_distribution_value(
                x_predict, parameter["mean"], parameter["covariance_inverse"],
                parameter["covariance_det"]
            )

            class_probability = class_likelihood * parameter["prior"]

            probabilites.append((parameter["label"], class_probability))

        max_probability = max(probabilites, key=lambda x: x[1])

        return max_probability[0], max_probability[1]

import math
import pickle

import numpy as np
import matplotlib.pyplot as plt


def generate_dataset():
    x = np.linspace(0, 1, 200)
    y = np.zeros_like(x, dtype=np.int32)

    x[0:100] = np.sin(4 * np.pi * x)[0:100]
    x[100:200] = np.cos(4 * np.pi * x)[100:200]

    y = 4 * np.linspace(0, 1, 200) + 0.3 * np.random.randn(200)

    label = np.ones_like(x)
    label[0:100] = 0

    with open("dataset.pkl", "wb") as file:
        pickle.dump((x, y, label), file)


def solve_quadratic_equation(a, b, c):
    b_square_minus_four_ac = pow(b, 2) - 4 * a * c

    if b_square_minus_four_ac < 0:
        raise Exception("Imaginary Value")
    else:
        y1 = (-b + math.sqrt(b_square_minus_four_ac)) / (2 * a)
        y2 = (-b - math.sqrt(b_square_minus_four_ac)) / (2 * a)

        return y1, y2


def gaussian_distribution_value(x, mu, sigma):
    """
    Computes the value of the point x having mean mu and covariance sigma
    """

    x_minus_mu = x - mu
    sigma_det = np.linalg.det(sigma)

    exponent = np.exp(
        -np.dot(
            np.dot(x_minus_mu.T, np.linalg.inv(sigma)),
            x_minus_mu
        ) / 2.0
    )

    value = exponent / (math.sqrt(pow(2 * math.pi, len(x)) * sigma_det))

    return value


class GaussianClassifier():

    def __init__(self, lda=False):
        self.lda = lda

    def fit(self, x, y, labels):
        # Store the training data used
        self.x_train = np.stack((x, y), axis=-1)
        self.y_train = labels

        self.number_of_training_data = len(self.x_train)

        # Obtain indices for classes
        class1_indices = np.where(self.y_train == 0)
        class2_indices = np.where(self.y_train == 1)

        # Calculate the priors for class 1 and 2
        self.class1_prior = len(
            class1_indices[0]) / self.number_of_training_data
        self.class2_prior = len(
            class2_indices[0]) / self.number_of_training_data

        # Obtain the data for both class 1 and class 2
        class1_data = self.x_train[class1_indices]
        class2_data = self.x_train[class2_indices]

        # Calculate the arithmetic mean for each class
        self.mu1 = np.mean(class1_data, axis=0)
        self.mu2 = np.mean(class2_data, axis=0)

        # Calculate the covariance matrices
        if(self.lda):
            self.covariance = np.cov(self.x_train, rowvar=False)
        else:
            self.covariance1 = np.cov(class1_data, rowvar=False)
            self.covariance2 = np.cov(class2_data, rowvar=False)

    def predict(self, x, y):
        x_predict = np.array([x, y])

        if(self.lda):
            class1_likelihood = gaussian_distribution_value(
                x_predict, self.mu1, self.covariance)
            class2_likelihood = gaussian_distribution_value(
                x_predict, self.mu2, self.covariance)
        else:
            class1_likelihood = gaussian_distribution_value(
                x_predict, self.mu1, self.covariance1)
            class2_likelihood = gaussian_distribution_value(
                x_predict, self.mu2, self.covariance2)

        class1_posteriori = class1_likelihood * self.class1_prior
        class2_posteriori = class2_likelihood * self.class2_prior

        if class1_posteriori > class2_posteriori:
            return 0
        else:
            return 1

    def plot(self):
        plt.scatter(*zip(*self.x_train), c=self.y_train)
        x = np.linspace(-2, 2, 1000)

        if(self.lda):
            # Calculate y
            y = -0.282 * x + 1.973

            decision_boundary, = plt.plot(
                x, y, label="LDA decision boundary")
        else:
            a = 2.092

            x_points = []
            y_points = []

            for x_point in x:
                b = 5.722 * x_point + 5.497
                c = (1.681 * (x_point ** 2)) - (6.081 * x_point) - 18.873

                try:
                    y1, y2 = solve_quadratic_equation(a, b, c)
                except:
                    continue

                x_points.append(x_point)
                y_points.append(y1)

            decision_boundary, = plt.plot(
                x_points,
                y_points,
                label="QDA decision boundary"
            )

        plt.legend(handles=[decision_boundary])
        plt.show()


if __name__ == "__main__":
    classifier = GaussianClassifier()

    with open("dataset.pkl", "rb") as file:
        dataset = pickle.load(file)

    classifier.fit(*dataset)

    print(classifier.predict(0.0, -0.30))
    print(classifier.predict(-0.99, 2.82))
    classifier.plot()

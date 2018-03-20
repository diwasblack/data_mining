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

    return x, y, label


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
            self.covariance_inverse = np.linalg.inv(self.covariance)
            self.covariance_det = np.linalg.det(self.covariance)
        else:
            self.covariance1 = np.cov(class1_data, rowvar=False)
            self.covariance1_inverse = np.linalg.inv(self.covariance1)
            self.covariance1_det = np.linalg.det(self.covariance1)

            self.covariance2 = np.cov(class2_data, rowvar=False)
            self.covariance2_inverse = np.linalg.inv(self.covariance2)
            self.covariance2_det = np.linalg.det(self.covariance2)

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

    def qda_decision_boundary(self, x, y):

        X = np.stack((x, y), axis=-1).reshape(-1, 2)

        def decision_function(X):
            X_minus_mu1 = X - self.mu1
            X_minus_mu2 = X - self.mu2

            d1 = np.dot(X_minus_mu1.T, np.dot(self.covariance1_inverse,
                                              X_minus_mu1)) - (1 / 2) * math.log(self.covariance1_det)

            d2 = np.dot(X_minus_mu2.T, np.dot(self.covariance2_inverse,
                                              X_minus_mu2)) - (1 / 2) * math.log(self.covariance2_det)

            diff = d1 - d2

            return diff

        values = np.apply_along_axis(decision_function, 1, X)

        return values.reshape(*x.shape)

    def plot(self):
        plt.clf()
        plt.scatter(*zip(*self.x_train), c=self.y_train)
        x = np.linspace(-1.5, 1.5, 1000)

        if(self.lda):
            covariance_inverse = np.linalg.inv(self.covariance)

            # Solve for slope of line
            m = np.dot(covariance_inverse, self.mu1 - self.mu2)

            # Solve for the intercept
            c1 = - np.dot(self.mu1.T,
                          np.dot(covariance_inverse, self.mu1)) / 2.0
            c2 = np.dot(self.mu2.T, np.dot(covariance_inverse, self.mu2)) / 2.0

            # Calculate the intercept
            c = c1 + c2

            # Calculate y
            y = - (m[0] / m[1]) * x - c / m[1]

            decision_boundary, = plt.plot(
                x, y, label="LDA decision boundary")
            plt.legend(handles=[decision_boundary])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig("lda.png")
        else:
            x = np.arange(-5, 1.5, 0.01)
            y = np.arange(-1.5, 5, 0.01)

            X, Y = np.meshgrid(x, y)

            decision_boundary = plt.contour(
                X, Y, self.qda_decision_boundary(X, Y), levels=[0], colors=("blue"))

            decision_boundary.collections[0].set_label("QDA decision boundary")
            plt.legend(loc="upper left")

            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig("qda.png")


if __name__ == "__main__":
    # Generate dataset to use
    print("Generating Dataset")
    dataset = generate_dataset()

    # Use LDA
    lda_classifier = GaussianClassifier(lda=True)
    print("Training LDA classifier")
    lda_classifier.fit(*dataset)
    print("Plotting decision boundary for LDA")
    lda_classifier.plot()

    # Use QDA
    qda_classifier = GaussianClassifier()
    print("Training QDA classifier")
    qda_classifier.fit(*dataset)
    print("Plotting decision boundary for QDA")
    qda_classifier.plot()

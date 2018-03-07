# -*- coding: utf-8 -*-
import logging

from sklearn.datasets import load_iris

import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data['data'], data['target'], random_state=0)


class KNN(object):
    def __init__(self):
        pass

    def train(self, X_train, y_train):
        """
        X = X_train
        y = y_train
        """
        self.X_train = X_train
        self.y_train = y_train

        self.labels = set(self.y_train)

    def predict_label(self, input_vector, k):

        distances = norm(np.array(input_vector) - self.X_train, axis=1)

        sorted_distances = sorted(
            zip(distances, self.y_train),
            key=lambda x: x[0]
        )

        # Obtain the k nearest neighbors
        k_nearest_neighbors = sorted_distances[:k]

        # Construct dictionary for storing class label count
        label_counts = {key: 0 for key in self.labels}

        # Perform majority voting for class label
        for neighbor in k_nearest_neighbors:
            neighbor_label = neighbor[1]
            label_counts[neighbor_label] = label_counts[neighbor_label] + 1

        label_counts_tuple = label_counts.items()
        sorted_label_counts = sorted(label_counts_tuple, key=lambda x: x[1],
                                     reverse=True)

        if sorted_label_counts[0][1] != sorted_label_counts[1][1]:
            return sorted_label_counts[0][0]
        else:
            return -1

    def predict(self, X_test, k=1):
        """
        It takes X_test as input, and return an array of integers, which are the
        class labels of the data corresponding to each row in X_test.
        Hence, y_project is an array of lables voted by their corresponding
        k nearest neighbors
        """

        class_iterator = (self.predict_label(x, k) for x in X_test)
        return np.fromiter(class_iterator, int)

    def report(self, X_test, y_test, k=1):
        """
        return the accurancy of the test data.
        """
        y_predicted = self.predict(X_test, k)
        comparison = np.equal(y_predicted, y_test)
        return np.count_nonzero(comparison) / len(X_test)


def k_validate(X_test, y_test):
    """
    plot the accuracy against k from 1 to a certain number so that one could pick the best k
    """
    # Create the KNN classifier
    knn = KNN()

    # Train the classifier
    knn.train(X_train, y_train)

    # Find accuracy for each value of K
    k_value_accuracies = [(i, knn.report(X_test, y_test, k=i))
                          for i in range(1, len(X_train))]

    # Find the k with highest accuracy
    best_k = max(k_value_accuracies, key=lambda x: x[1])

    logger.info("The highest accuracy {} is obtained for k={}".format(
        best_k[1], best_k[0]))

    plt.plot(*zip(*k_value_accuracies))

    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.savefig("validation_accuracy.png")

    return best_k


if __name__ == "__main__":
    k_validate(X_test, y_test)

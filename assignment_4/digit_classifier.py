import math
import logging

import numpy as np
import cloudpickle as pickle


def sigmoid_function(z):
    """
    Evaluate sigmoid function at point z
    """

    if(z >= 500):
        return 0.9999
    elif(z <= -500):
        return 0.0001
    else:
        value = 1.0 / (1 + math.exp(-1 * z))

    return value


def train_test_split(x, y, test_size=0.33):
    """
    Helper function to split the data into training set and test set
    """
    data_size = len(x)

    # Generate random permutation of data set
    p = np.random.permutation(data_size)

    # Shuffle x and y
    x = x[p]
    y = y[p]

    training_size = int((1 - test_size) * data_size)

    x_train = x[:training_size]
    x_test = x[training_size:]

    y_train = y[:training_size]
    y_test = y[training_size:]

    return x_train, x_test, y_train, y_test


def compute_accuracy(y_test, y_predicted):
    """
    Compute the accuracy of the classification labels
    """

    data_size = y_test.shape[0]

    tp_tn_sum = 0

    for y1, y2 in zip(y_test, y_predicted):
        if(y1 == y2):
            tp_tn_sum += 1

    return tp_tn_sum / data_size


class LogisticRegression():
    def __init__(self, learning_rate=0.05, threshold=0.5):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.sigmoid_vectorizer = np.vectorize(sigmoid_function)

    def fit(self, x_train, y_train):
        self.input_dimension = x_train.shape[1]
        self.number_of_training_data = x_train.shape[0]

        # Obtain the labels from training set
        self.labels = list(set(y_train))

        if(len(self.labels) != 2):
            raise Exception(
                "Received multiple labels for binary classification")

        # Convert labels to binary values
        y_train = np.array([self.labels.index(x) for x in y_train])
        y = y_train.reshape(self.number_of_training_data, -1)

        # Bias terms to add
        bias = np.ones((self.number_of_training_data, 1))

        # Augmented training data
        x = np.hstack((bias, x_train))

        # Randomly initialize thetas
        self.thetas = np.random.uniform(-0.5,
                                        0.5, (1, self.input_dimension + 1))

        logging.info("Training classifier")

        while(True):
            z = np.dot(x, self.thetas.T)

            predicted_values = self.sigmoid_vectorizer(z)

            error_signal = y - predicted_values

            theta_updates = (1 / self.number_of_training_data) * \
                np.dot(error_signal.T, x)

            # Perform batch update on the data
            self.thetas = self.thetas + self.learning_rate * theta_updates

            max_update = np.max(np.abs(theta_updates))
            logging.info("Max correction for theta: {}".format((max_update)))

            if(max_update <= self.threshold):
                break

    def predict(self, x_predict):
        data_size = x_predict.shape[0]
        bias = np.ones((data_size, 1))
        x = np.hstack((bias, x_predict))

        z = np.dot(x, self.thetas.T)
        values = self.sigmoid_vectorizer(z)

        predicted_class = np.array(
            [1 if value > 0.5 else 0 for value in values])

        # Convert binary values to labels
        predicted_labels = [self.labels[x] for x in predicted_class]

        return predicted_labels


def train_and_test_classifier(x_train, y_train, x_test, y_test):
    """
    Helper function to train the classifier and obtain its accuracy
    """

    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    classifier = LogisticRegression()

    classifier.fit(x_train, y_train)
    y_predicted = classifier.predict(x_test)
    accuracy = compute_accuracy(y_test, y_predicted)

    logger.info("Accuracy: {}".format(accuracy))


def main():

    mnist23 = pickle.load(open("mnist23.data", "rb"))

    x = mnist23.data
    y = mnist23.target

    # Perform train test split on the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    train_and_test_classifier(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()

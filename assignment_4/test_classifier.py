import logging

import numpy as np
import cloudpickle as pickle

from sklearn.model_selection import train_test_split

from logistic_regression import LogisticRegression

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    mnist23 = pickle.load(open("mnist23.data", "rb"))

    x = mnist23.data
    y = np.array([1 if x == 3 else 0 for x in mnist23.target])

    classifier = LogisticRegression()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    classifier.fit(x_train, y_train)
    accuracy = classifier.compute_accuracy(x_test, y_test)

    logger.info("Accuracy: {}".format(accuracy))


if __name__ == "__main__":
    main()

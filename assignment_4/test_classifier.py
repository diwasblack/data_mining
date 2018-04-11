import logging

import numpy as np
import cloudpickle as pickle

from sklearn.metrics import accuracy_score
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

    y_predicted = classifier.predict(x_test)

    logger.info("Accuracy: {}".format(accuracy_score(y_test, y_predicted)))


if __name__ == "__main__":
    main()

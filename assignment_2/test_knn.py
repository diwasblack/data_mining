import unittest
import logging
import time

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from knn import KNN

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestKNN(unittest.TestCase):

    def setUp(self):
        self.knn = KNN()

        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            data['data'], data['target'], random_state=0)

        self.knn.train(X_train, y_train)
        self.X_test = X_test
        self.y_test = y_test

    def test_distance_measure(self):
        a = np.random.rand(5)
        b = np.random.rand(5)

        self.assertTrue(self.knn.measure_distance(a, b))

    def test_prediction_time(self):
        start_time = time.time()
        self.knn.predict(self.X_test, k=25)
        end_time = time.time()
        diff = (end_time - start_time) * 1000
        logger.info("Running time for K={} is: {}".format(25, diff))


if __name__ == '__main__':
    unittest.main()

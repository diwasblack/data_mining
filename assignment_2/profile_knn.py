from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from knn import KNN

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data['data'], data['target'], random_state=0)

knn = KNN()
knn.train(X_train, y_train)

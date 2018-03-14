Data Mining
===========

Assignment 1
------------
1. Dice-Model: Suppose you have a 6-sided dice. You throw the dice for n times altogether.
In the end, you record ni  times for face i. In this question, you are asked to derive the probability P(face =i) for each face I, by using maximum-likelihood.
2. Let G be a transitional probability matrix, in which every entry is nonnegative,
and the sum of all the entries for every column is 1. As discussed in class, this matrix has an
engivector for the engivalue one, which can be viewed as pageranking for all the nodes of the graph.
In this question, you are asked to implement a function in python to compute such an engivector for the given parameter matrix G.
You should use arrays in numpy, functions as abs() could be helpful when implementing your code. There are many ways to implement such a function.
For example, you could use the so-called power method to iteratively compute such an engivector, and to halt the computation when the error is
less than a certain prechosen threshold (eg., 1e-5). You could use abs() to measure the error.

Name your function as pageranking, which takes G as the input parameter and returns the pageranking verctor.


Assignment 2
------------
In this homework, you will implement a K-nearest neighbor classifier, which probably is the simplest kind of classification model.
The KNN model is a simple non-parameterized classification model. So in the training part,
the model simply remembers all the training data and their corresponding class labels. In the predict part, the model simply
computes the distance of the input data to the training data and choose k nearest neighbors. The simple majority vote from the
k closest neighbors is the class label for the input data.

Issue: If there is no simple majority vote, (i.e., two or more class labels receive the same maximum votes, then we do not classify the input)

Hyperparameter(s): k is a hyperparameter. Since k is a hyper parameter, we need to use the X_test to find the best k. Before you do that, you need to plot the accuracy against k.

Prepare the following skeleton program for you and you are asked to:
1. complete the skeleton
2. time your predict() function.
3. Plot the accuracy against k
4. Pick the best k for the highest accuracy. Let’s denote it as x%.
5. The answer the following question: can we report the x% accuracy as the accuracy to users of the program. Why or why not?


Assignment 3
------------
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,200)
y = np.zeros_like(x,dtype = np.int32)

x[0:100] = np.sin(4*np.pi*x)[0:100]
x[100:200] = np.cos(4*np.pi*x)[100:200]

y = 4*np.linspace(0,1,200)+0.3*np.random.randn(200)
label= np.ones_like(x)
label[0:100]=0
plt.scatter(x,y,c=label)
```

Run the above python file to generate data. The data sets contain 200 data
points and they belong to 2 classes, where the first 100 data points are labeled as
class 0 and the second data points are labeled as class 1. Apply LDA and QDA to
data set to find its decision boundary respectively.  Plot your decision
boundaries in your answer sheet as well.  Note that, when generating y, a
random normalized noise is added with a 0.3 factor. You can change the value
to see how your decision boundary might be affected.

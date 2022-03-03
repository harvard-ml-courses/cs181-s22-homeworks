import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt


# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.errors = []

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # Append ones to each data point
    def __basis(self, X):
        return np.stack((np.array([np.ones(len(X))])[0], X[:,0], X[:,1])).T

    def __gradient(self, X, y):
        N = len(y)
        K = 3
        losses = [] # different gradient for each class k
        for k in range(K):
            loss = 0
            for n in range(N):
                predict = softmax(np.matmul(self.W, X[n]))
                loss += (predict[k] - y[n][k]) * X[n]
            losses.append(loss)
        regularization = 2 * self.lam * self.W
        final = np.concatenate(losses).reshape(3, 3) + np.array(regularization)
        # cross entropy loss + regularization
        return final

    def __error(self, X, y):
        error = 0
        N = len(y)
        K = 3
        for n in range(N):
            for k in range(K):
                predict = softmax(np.matmul(self.W, X[n]))
                error -= y[n][k] * np.log(predict[k]) # for visualize_loss function
        self.errors.append(error)

    # TODO: Implement this method!
    def fit(self, X, y):
        X_transformed = self.__basis(X) # transform X to include bias
        y_hot = [] # encode y as one-hot vectors
        for y_class in y:
            one_hot = [0, 0, 0]
            one_hot[y_class] = 1
            y_hot.append(one_hot)
        y_hot = np.array(y_hot)
        
        self.W = np.random.randn(X_transformed.shape[1], 3) # initialize random weights
        runs = 200000

        # gradient descent to update weights
        for r in range(runs):
            print("Logistic Regression: Run #", r)
            self.W = self.W - self.eta * self.__gradient(X_transformed, y_hot)
            self.__error(X_transformed, y_hot)

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        X_pred_transformed = self.__basis(X_pred)
        for x in X_pred_transformed:
            y_hat = softmax(np.matmul(self.W, x)) # take softmax of predicted values
            y_max = np.argmax(y_hat) # most likely class
            preds.append(y_max)
        return np.array(preds)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        print(self.errors)
        plt.figure()
        plt.title('Loss vs. Iterations for Logistic Regression')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Negative Log Likelihood')
        plt.plot(list(range(200000)), self.errors, '.-')
        return
    # plot loss vs. number of iterations

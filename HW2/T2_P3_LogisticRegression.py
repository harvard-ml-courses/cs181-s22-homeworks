import numpy as np



# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.runs = 200000

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __softmax(self, z):
        softmaxes = []
        for row in z.T:
            softmax_row = []
            denominator = 0
            for z_n in row:
                denominator += pow(np.e, z_n)
            for z_n in row:
                softmax_row.append(pow(np.e, z_n) / denominator)
            softmaxes.append(softmax_row)
        return np.array(softmaxes).T

    def __gradient(self, X, y):
        # gradients = np.zeros(self.W.shape) # 3 x 3
        
        one_hot_matrix = np.zeros((3, X.shape[0])) # 3 x 27
        for i, elt in enumerate(y):
            one_hot_matrix[elt][i] = 1

        classification_probs = self.__softmax(np.dot(self.W, X.T)) # 3 x 27

        gradients = np.dot((classification_probs - one_hot_matrix), X) # 3 x 3

        reg_gradients = gradients + 2 * self.lam * self.W # 3 x 3

        # for i, row in enumerate(classification_probs):
        #     real_probs = np.array(np.zeros(3))
        #     real_probs[y[i]] = 1
        #     gradients += (row - real_probs) * X[i]
        # reg_gradients = gradients + 2 * self.lam * self.W
        return reg_gradients

    # TODO: Implement this method!
    def fit(self, X, y, w_init=None):
        # Add column of 1s as bias term in X matrix
        X = np.array(X)
        X = np.hstack((np.ones((X.shape[0], 1)), X)) # 27 x 3

        if w_init is not None:
            self.W = w_init
        else:
            self.W = np.random.rand(3, X.shape[1]) # 3 x 3
        
        gradients = self.__gradient(X, y) # 3 x 3
        for _ in range(self.runs):
            self.W = self.W - self.eta * gradients
            print(self.W)
            gradients = self.__gradient(X, y)

    # TODO: Implement this method!
    def predict(self, X_pred):
        X_pred = np.hstack((np.ones((X_pred.shape[0], 1)), X_pred))
        predictions = []
        classification_probs = self.__softmax(np.dot(self.W, X_pred.T))
        for row in classification_probs.T:
            predictions.append(np.argmax(row))
        
        return np.array(predictions)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        pass

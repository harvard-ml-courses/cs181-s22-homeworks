from functools import total_ordering
import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance
        
        self.mean_matrix = np.zeros((3,2))
        
        self.cov_matrix = np.zeros((2,2))
        self.cov_matrix0 = np.zeros((2,2))
        self.cov_matrix1 = np.zeros((2,2))
        self.cov_matrix2 = np.zeros((2,2))

        self.num_class0 = 0
        self.num_class1 = 0
        self.num_class2 = 0
        self.total = 0

        self.X_0 = []
        self.X_1 = []
        self.X_2 = []

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):
        # X is 27 x 2
        magnitudes = (X.T)[0]
        temperatures = (X.T)[1]

        self.num_class0 = np.count_nonzero(y == 0)
        self.num_class1 = np.count_nonzero(y == 1)
        self.num_class2 = np.count_nonzero(y == 2)
        self.total = (X.shape)[0]

        mag0 = 0
        temp0 = 0
        mag1 = 0
        temp1 = 0
        mag2 = 0
        temp2 = 0

        for i, y_n in enumerate(y):
            if y_n == 0:
                mag0 += magnitudes[i]
                temp0 += temperatures[i]
                self.X_0.append(X[i])
            if y_n == 1:
                mag1 += magnitudes[i]
                temp1 += temperatures[i]
                self.X_1.append(X[i])
            if y_n == 2:
                mag2 += magnitudes[i]
                temp2 += temperatures[i]
                self.X_2.append(X[i])
        
        av_mag0 = mag0 / self.num_class0
        av_mag1 = mag1 / self.num_class1
        av_mag2 = mag2 / self.num_class2
        av_temp0 = temp0 / self.num_class0
        av_temp1 = temp1 / self.num_class1
        av_temp2 = temp2 / self.num_class2

        self.mean_matrix = np.array([[av_mag0, av_temp0], [av_mag1, av_temp1], [av_mag2, av_temp2]]) # 3 x 2

        # 2 x 2
        self.cov_matrix0 = np.cov(np.array(self.X_0).T)
        self.cov_matrix1 = np.cov(np.array(self.X_1).T)
        self.cov_matrix2 = np.cov(np.array(self.X_2).T)

        self.cov_matrix = ((self.cov_matrix0 * self.num_class0) + (self.cov_matrix1 * self.num_class1) + (self.cov_matrix2 * self.num_class2)) / self.total

        return 

    # TODO: Implement this method!
    def predict(self, X_pred):
           
        predictions = []

        for row in X_pred:
            classification_probs = []

            if self.is_shared_covariance:
                prob_class0 = mvn.pdf(row, self.mean_matrix[0], self.cov_matrix) * self.num_class0 / self.total
                prob_class1 = mvn.pdf(row, self.mean_matrix[1], self.cov_matrix) * self.num_class1 / self.total
                prob_class2 = mvn.pdf(row, self.mean_matrix[2], self.cov_matrix) * self.num_class2 / self.total
            else:
                prob_class0 = mvn.pdf(row, self.mean_matrix[0], self.cov_matrix0) * self.num_class0 / self.total
                prob_class1 = mvn.pdf(row, self.mean_matrix[1], self.cov_matrix1) * self.num_class1 / self.total
                prob_class2 = mvn.pdf(row, self.mean_matrix[2], self.cov_matrix2) * self.num_class2 / self.total
            classification_probs = [prob_class0, prob_class1, prob_class2]
            
            predictions.append(np.argmax(classification_probs))
        
        return np.array(predictions)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        loss0 = 0
        loss1 = 0
        loss2 = 0

        if self.is_shared_covariance:
            for row in self.X_0:
                prob_x = mvn.pdf(row, self.mean_matrix[0], self.cov_matrix)
                loss0 += np.log(prob_x) + np.log(self.num_class0 / self.total)
            for row in self.X_1:
                prob_x = mvn.pdf(row, self.mean_matrix[1], self.cov_matrix)
                loss1 += np.log(prob_x) + np.log(self.num_class1 / self.total)
            for row in self.X_2:
                prob_x = mvn.pdf(row, self.mean_matrix[2], self.cov_matrix)
                loss2 += np.log(prob_x) + np.log(self.num_class2 / self.total)
        else:
            for row in self.X_0:
                prob_x = mvn.pdf(row, self.mean_matrix[0], self.cov_matrix0)
                loss0 += np.log(prob_x) + np.log(self.num_class0 / self.total)
            for row in self.X_1:
                prob_x = mvn.pdf(row, self.mean_matrix[1], self.cov_matrix1)
                loss1 += np.log(prob_x) + np.log(self.num_class1 / self.total)
            for row in self.X_2:
                prob_x = mvn.pdf(row, self.mean_matrix[2], self.cov_matrix2)
                loss2 += np.log(prob_x) + np.log(self.num_class2 / self.total)
        
        return loss0 + loss1 + loss2



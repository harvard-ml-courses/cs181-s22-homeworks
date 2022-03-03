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

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):
        N = len(y)
        K = 3

        # Means
        self.mus = []
        for k in range(K):
            N_k = list(y).count(k)
            mu = 0
            for n in range(N):
                if y[n] == k:
                    mu += 1 / N_k * X[n]
            self.mus.append(mu)
        #print("MUs", self.mus, "\n")

        # Priors
        self.priors = []
        for k in range(K):
            self.priors.append(list(y).count(k) / N)
        #print("PRIORS", self.priors, "\n")

        # Shared covariance matrix
        self.covs = []
        if self.is_shared_covariance:
            cov = np.zeros([2, 2])
            for n in range(N):
                for k in range(K):
                    if y[n] == k:
                        cov += 1/N * np.matmul(np.array([X[n]-self.mus[k]]).T, np.array([X[n]-self.mus[k]]))
            for k in range(K):
                self.covs.append(cov)

        # Separate covariance matrix
        else:
            for k in range(K):
                cov = np.zeros([2, 2])
                for n in range(N):
                    if y[n] == k:
                        cov += 1/(list(y).count(k)) * np.matmul(np.array([X[n]-self.mus[k]]).T, np.array([X[n]-self.mus[k]]))
                self.covs.append(cov)

        #print("covariance of class 0\n", self.covs[0], "\n")
        #print("covariance of class 1\n", self.covs[1], "\n")
        #print("covariance of class 2\n", self.covs[2], "\n")

        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        K = 3
        preds = []
        for x in X_pred:
            likelihoods = []
            # find likelihood for each class
            for k in range(K):
                class_conditional = mvn.pdf(x, mean=self.mus[k], cov=self.covs[k])
                likelihoods.append(class_conditional * self.priors[k])
            max_index = np.argmax(likelihoods)
            preds.append(max_index) # prediction is highest likelihood
        return np.array(preds)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        N = len(y)
        K = 3

        likelihood = 0
        for n in range(N):
            for k in range(K):
                if y[n] == k:
                    class_conditional = mvn.pdf(X[n], mean=self.mus[k], cov=self.covs[k])
                    likelihood += np.log(class_conditional) + np.log(self.priors[k])
                    # likelihood += 1/2 * (X[n]-self.mus[k]).T * np.linalg.inv(self.covs[k]) * (X[n] - self.mus[k]) + np.log(1/np.sqrt(np.linalg.det(2 * np.pi * self.covs[k]))) + np.log(self.priors[k])
        
        return -1 * likelihood
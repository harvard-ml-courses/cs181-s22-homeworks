import numpy as np

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __dist(self, star1, star2):
        return ((star1[0] - star2[0])/3) ** 2 + (star1[1] - star2[1]) ** 2

    # TODO: Implement this method!
    def predict(self, X_pred):
        preds = []
        for x_new in X_pred:

            # find distances between all neighbors of x_new
            dists = []
            for x_old in self.X:
                star_dist = self.__dist(x_old, x_new)
                if star_dist > 0:
                    dists.append(star_dist)
                else: # make sure testing point isn't same as training point
                    dists.append(float('inf'))

            # sort distances and choose k smallest
            sorted_dists = sorted(dists)
            knn_dists = sorted_dists[0:self.K]

            # find k indices that correspond to k smallest distances
            knn_indices = []
            for n in knn_dists:
                knn_index = dists.index(n)
                knn_indices.append(knn_index)
                dists[knn_index] = -1 # make sure same index isn't chosen twice

            # find y that correspond to indices of knn
            knn_y = [self.y[i] for i in knn_indices]
            preds.append(max(set(knn_y), key=knn_y.count))
        return np.array(preds)

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y
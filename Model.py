import numpy as np
from Data import Data

class Model:
    def __init__(self):
        pass

    # call the train method from Data.py to get training data
    def train(self):
        X_train, Y_train , X_test, Y_test= Data.load_data()
        parameters, costs = Data.train(X_train, Y_train)
        self.w, self.b = parameters['w'], parameters['b']

        return X_train, Y_train, X_test, Y_test

    # Predict method to make predictions on new data
    def predict(self, x):
        m = x.shape[1]
        Y_prediction = np.zeros((1, m))
        w = self.w.reshape(x.shape[0], 1)

        A = Data.sigmoid(np.dot(w.T, x) + self.b)

        for i in range(A.shape[1]):
            if A[0, i] > 0.5:
                Y_prediction[0, i] = 1
            else:
                Y_prediction[0, i] = 0

        return Y_prediction

    # Evaluate method to calculate accuracy
    def evaluate(self, X, Y):
        predictions = self.predict(X)
        # For accuracy, consider only definite predictions (0 or 1)
        valid_mask = (predictions == 0) | (predictions == 1)
        if np.sum(valid_mask) == 0:
            return 0.0
        correct = np.sum((predictions == Y) & valid_mask)
        accuracy = correct / np.sum(valid_mask) * 100
        return accuracy


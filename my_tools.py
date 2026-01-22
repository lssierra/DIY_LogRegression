import numpy as np

class my_LogRegression():
    def __init__(self):
        self.beta_ = None
        self.iterations = 100
        self.epsilon = 1e-7
        self.learnr = 0.1
        self.threshhold = 0.5




    def Sigmoid(self, alpha):
        return 1.0/ (1 + np.exp(-alpha))

    def fit(self, X, y):
        m = y.size
        Xp = np.c_[np.ones((X.shape[0],1)),X]
        self.beta_ = np.zeros(Xp.shape[1])

        for i in range(self.iterations):
            grad_beta = (Xp.T @ (self.Sigmoid(Xp @ self.beta_) - y))/m
            self.beta_ -= self.learnr * grad_beta

            if np.linalg.norm(grad_beta) < self.epsilon:
                break

    def Predict_probas(self, X):
        Xp = np.c_[np.ones((X.shape[0],1)),X]
        return self.Sigmoid(Xp @ self.beta_)

    def Predict(self, X):
        probas = self.Predict_probas(X)
        return (probas >= self.threshhold).astype(int)
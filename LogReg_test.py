import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def Sigmoid(alpha):
    return 1.0/ (1 + np.exp(-alpha))



def fit(X, y):
    m = y.size
    Xp = np.c_[np.ones((X.shape[0],1)),X]
    beta_ = np.zeros(Xp.shape[1])

    for i in range(100):
        grad_beta = (Xp.T @ (Sigmoid(Xp @ beta_) - y))/m
        beta_ -= 0.1 * grad_beta

        if np.linalg.norm(grad_beta) < 1e-7:
            break
    return beta_


def Predict_probas(X, beta_):
    Xp = np.c_[np.ones((X.shape[0],1)),X]
    return Sigmoid(Xp @ beta_)

def Predict(X, beta_):
    probas = Predict_probas(X, beta_)
    return (probas >= 0.5).astype(int)


X = np.random.rand(200, 2) * 10
y = ((X[:,0] + X[:,1]) > 10).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


beta_hat = fit(X_train,y_train)

y_predTrain = Predict(X_train,beta_hat)
y_predTest = Predict(X_test,beta_hat)


print("train accuracy = ",accuracy_score(y_train,y_predTrain))
print("test accuracy = ",accuracy_score(y_test,y_predTest))

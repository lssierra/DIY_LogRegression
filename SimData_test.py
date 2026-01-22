import numpy as np
import matplotlib.pyplot as plt


def Gen_morethan10():
    X = np.random.rand(200, 2) * 10
    y = ((X[:,0] + X[:,1]) > 10).astype(int)
    return X,y

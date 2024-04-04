import numpy as np
import torch
from torch import nn
import numpy as np
from sklearn.linear_model import Ridge
import helpers
import matplotlib.pyplot as plt


class Ridge_Regression:

    def __init__(self, lambd):
        self.lambd = lambd
        self.W = None

    def fit(self, X, Y):

        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """

        Y = 2 * (Y - 0.5) # transform the labels to -1 and 1, instead of 0 and 1.
        
        ########## YOUR CODE HERE ##########

        # compute the ridge regression weights using the formula from class / exercise.
        # you may not use np.linalg.solve, but you may use np.linalg.inv

        ####################################
        #ridge_classifier = Ridge(alpha=self.lambd, fit_intercept=False, solver='svd')
        N_train = X.shape[0]
        self.W = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)/N_train + self.lambd * np.eye(X.shape[1])), X.T), Y/N_train)
        #ridge_classifier.fit(X, Y)
        #self.W = ridge_classifier.coef_
        
        


    def predict(self, X):
        """
        Predict the output for the provided data.
        :param X: The data to predict. np.ndarray of shape (N, D).
        :return: The predicted output. np.ndarray of shape (N,), of 0s and 1s.
        """
        ########## YOUR CODE HERE ##########

        # compute the predicted output of the model.
        # name your predicitons array preds.

        ####################################

        # transform the labels to 0s and 1s, instead of -1s and 1s.
        # You may remove this line if your code already outputs 0s and 1s.
        preds =  np.where(np.dot(X, self.W) >= 0, 1, -1) 
        preds = (preds + 1) / 2
        return preds      



class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic_Regression, self).__init__()

        ########## YOUR CODE HERE ##########

        # define a linear operation.

        ####################################
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Computes the output of the linear operator.
        :param x: The input to the linear operator.
        :return: The transformed input.
        """
        # compute the output of the linear operator

        ########## YOUR CODE HERE ##########

        # return the transformed input.
        # first perform the linear operation
        # should be a single line of code.

        ####################################

        return self.linear(x)

    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x

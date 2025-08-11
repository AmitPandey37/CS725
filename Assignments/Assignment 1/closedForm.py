import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(2024)

class LinearRegressionClosedForm:
    def __init__(self):
        '''
        Initializing the parameters of the model

        Returns:
            None
        '''
        self.weights = None

    def fit(self, X, y):
        '''
        This function is used to obtain the weights of the model using closed form solution.

        Args:
            X : 2D numpy array of training set data points. Dimensions (n x (d+1))
            y : 2D numpy array of target values in the training dataset. Dimensions (n x 1)

        Returns :
            None
        '''
        # Closed-form solution to linear regression using np.dot for matrix multiplication
        self.weights = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    def predict(self, X):
        '''
        This function is used to predict the target values for the given set of feature values

        Args:
            X: 2D numpy array of data points. Dimensions (n x (d+1))

        Returns:
            2D numpy array of predicted target values. Dimensions (n x 1)
        '''
        return np.dot(X, self.weights)

def plot_learned_equation(X, y, y_hat):
    '''
    This function generates the plot to visualize how well the learned linear equation fits the dataset  

    Args:
        X : 2D numpy array of data points. Dimensions (n x 2)
        y : 2D numpy array of target values. Dimensions (n x 1)
        y_hat : 2D numpy array of predicted values. Dimensions (n x 1)

    Returns:
        None
    '''
    # Plotting the actual data points and the predicted line
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 1], y, color='blue', label='Actual data')
    plt.plot(X[:, 1], y_hat, color='red', label='Predicted line')
    plt.title('Closed-form Solution of Linear Regression')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.savefig('closed_form.png')
    plt.show()

############################################
#####        Helper functions          #####
############################################
def generate_toy_dataset():
    '''
    This function generates a simple toy dataset containing 300 points with 1d feature 
    '''
    X = np.random.rand(300, 2)
    X[:, 0] = 1  # bias term
    weights = np.random.rand(2,1)
    noise = np.random.rand(300,1) / 32
    y = np.dot(X, weights) + noise
    
    X_train = X[:250]
    X_test = X[250:]
    y_train = y[:250]
    y_test = y[250:]
    return X_train, y_train, X_test, y_test

# Terminal text coloring
RESET = '\033[0m'
GREEN = '\033[32m'
RED = '\033[31m'

if __name__ == '__main__':
    
    print(RED + "##### Closed form solution for linear regression #####")
    
    print(RESET +  "Loading dataset: ", end="")
    try:
        X_train, y_train, X_test, y_test = generate_toy_dataset()
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Calculating closed form solution: ", end="")
    try:
        linear_reg = LinearRegressionClosedForm()
        linear_reg.fit(X_train, y_train)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Predicting for test split: ", end="")
    try:
        y_hat = linear_reg.predict(X_test)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Plotting the solution: ", end="")
    try:
        plot_learned_equation(X_test, y_test, y_hat)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Activation Functions and Their Derivatives

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)  # Vectorized for numpy arrays

def relu_derivative(x):
    """Derivative of ReLU."""
    return (x > 0).astype(int)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid."""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh."""
    return 1 - np.tanh(x) ** 2

# Mapping from string to activation functions and their derivatives
str_to_func = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative)
}

def get_activation_functions(activations):  
    """
    Given a list of activation function names, return the corresponding functions and their derivatives.
    
    Parameters:
    activations (list of str): List of activation function names.
    
    Returns:
    tuple: Two lists containing activation functions and their derivatives.
    """
    activation_funcs, activation_derivatives = [], []
    for activation in activations:
        activation_func, activation_derivative = str_to_func[activation]
        activation_funcs.append(activation_func)
        activation_derivatives.append(activation_derivative)
    return activation_funcs, activation_derivatives

# Neural Network Class
class NN:
    def __init__(self, input_dim, hidden_dims, activations=None):
        '''
        Initialize the Neural Network.
        
        Parameters
        ----------
        input_dim : int
            Size of the input layer.
        hidden_dims : list of int
            List containing the number of neurons in each hidden layer.
        activations : list of str, optional
            List containing activation function names for each hidden layer.
            If None, sigmoid activation is used for all layers.
        '''
        assert(len(hidden_dims) > 0), "There must be at least one hidden layer."
        assert(activations is None or len(hidden_dims) == len(activations)), \
            "Number of activations must match number of hidden layers."
         
        
        if activations is None:
            self.activations = [sigmoid] * (len(hidden_dims) + 1)
            self.activation_derivatives = [sigmoid_derivative] * (len(hidden_dims) + 1)
        else:
            self.activations, self.activation_derivatives = get_activation_functions(activations + ['sigmoid'])

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        layer_sizes = [input_dim] + hidden_dims + [1]  
        for i in range(len(layer_sizes) - 1):
            # He Initialization for weights
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1. / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        '''
        Perform the forward pass.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (N, D).
        
        Returns
        -------
        numpy.ndarray
            Output probabilities of shape (N, 1).
        '''
        self.Zs = []  
        self.As = [X]  
        for i in range(len(self.weights)):
            W = self.weights[i]
            b = self.biases[i]
            activation_func = self.activations[i]
            Z = self.As[-1] @ W + b 
            A = activation_func(Z)  
            self.Zs.append(Z)
            self.As.append(A)
        output_probs = self.As[-1]
        return output_probs

    def backward(self, X, y):
        '''
        Perform the backward pass (backpropagation).
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (N, D).
        y : numpy.ndarray
            Target labels of shape (N, 1).
        
        Returns
        -------
        tuple
            Gradients of weights and biases.
        '''
        m = y.shape[0]
        self.grad_weights = [np.zeros_like(W) for W in self.weights]
        self.grad_biases = [np.zeros_like(b) for b in self.biases]
        y = y.reshape(-1, 1)

        # Compute gradient for output layer
        A_output = self.As[-1]
        delta = (A_output - y) / m  

        # Gradient for the last layer
        A_prev = self.As[-2]
        dW = A_prev.T @ delta
        db = np.sum(delta, axis=0, keepdims=True)
        self.grad_weights[-1] = dW
        self.grad_biases[-1] = db

        # Backpropagate through hidden layers
        for l in reversed(range(len(self.weights) - 1)):
            W_next = self.weights[l + 1]
            delta = (delta @ W_next.T) * self.activation_derivatives[l + 1](self.Zs[l])
            A_prev = self.As[l]
            dW = A_prev.T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            self.grad_weights[l] = dW
            self.grad_biases[l] = db

        return self.grad_weights, self.grad_biases

    def step_bgd(self, weights, biases, delta_weights, delta_biases, optimizer_params, epoch):
        '''
        Update weights and biases using Batch Gradient Descent.
        
        Parameters
        ----------
            weights: list of numpy.ndarray
                Current weights of the network.
            biases: list of numpy.ndarray
                Current biases of the network.
            delta_weights: list of numpy.ndarray
                Gradients of weights with respect to loss.
            delta_biases: list of numpy.ndarray
                Gradients of biases with respect to loss.
            optimizer_params: dict
                Dictionary containing optimizer parameters:
                    - learning_rate: float
                    - gd_flag: int (1: Vanilla GD, 2: GD with Exponential Decay, 3: Momentum)
                    - momentum: float (used if gd_flag == 3)
                    - decay_constant: float (used if gd_flag == 2)
            epoch: int
                Current epoch number.
        
        Returns
        -------
        tuple
            Updated weights and biases.
        '''
        gd_flag = optimizer_params['gd_flag']
        learning_rate = optimizer_params['learning_rate']
        momentum = optimizer_params.get('momentum', 0.9)
        decay_constant = optimizer_params.get('decay_constant', 0.0)

        updated_W = []
        updated_B = []

        if gd_flag == 1:  # Vanilla GD
            for w, dw in zip(weights, delta_weights):
                updated_W.append(w - learning_rate * dw)
            for b, db in zip(biases, delta_biases):
                updated_B.append(b - learning_rate * db)

        elif gd_flag == 2:  # GD with Exponential Learning Rate Decay
            current_lr = learning_rate * np.exp(-decay_constant * epoch)
            for w, dw in zip(weights, delta_weights):
                updated_W.append(w - current_lr * dw)
            for b, db in zip(biases, delta_biases):
                updated_B.append(b - current_lr * db)

        elif gd_flag == 3:  # GD with Momentum
            if not hasattr(self, 'velocity_W'):
                # Initialize velocities
                self.velocity_W = [np.zeros_like(dw) for dw in delta_weights]
                self.velocity_B = [np.zeros_like(db) for db in delta_biases]
            for i, (w, dw) in enumerate(zip(weights, delta_weights)):
                self.velocity_W[i] = momentum * self.velocity_W[i] - learning_rate * dw
                updated_W.append(w + self.velocity_W[i])
            for i, (b, db) in enumerate(zip(biases, delta_biases)):
                self.velocity_B[i] = momentum * self.velocity_B[i] - learning_rate * db
                updated_B.append(b + self.velocity_B[i])
        else:
            raise ValueError(f"Invalid gd_flag: {gd_flag}")

        return updated_W, updated_B

    def step_adam(self, weights, biases, delta_weights, delta_biases, optimizer_params):
        '''
        Update weights and biases using Adam optimizer.
        
        Parameters
        ----------
            weights: list of numpy.ndarray
                Current weights of the network.
            biases: list of numpy.ndarray
                Current biases of the network.
            delta_weights: list of numpy.ndarray
                Gradients of weights with respect to loss.
            delta_biases: list of numpy.ndarray
                Gradients of biases with respect to loss.
            optimizer_params: dict
                Dictionary containing optimizer parameters:
                    - learning_rate: float
                    - beta1: float
                    - beta2: float
                    - eps: float
        
        Returns
        -------
        tuple
            Updated weights and biases.
        '''
        learning_rate = optimizer_params['learning_rate']
        beta1 = optimizer_params['beta1']
        beta2 = optimizer_params['beta2']
        eps = optimizer_params['eps']       

        if not hasattr(self, 't'):
            self.t = 0
            self.m_W = [np.zeros_like(dw) for dw in delta_weights]
            self.v_W = [np.zeros_like(dw) for dw in delta_weights]
            self.m_B = [np.zeros_like(db) for db in delta_biases]
            self.v_B = [np.zeros_like(db) for db in delta_biases]

        self.t +=1
        updated_W = []
        updated_B = []

        for i, (w, dw) in enumerate(zip(weights, delta_weights)):
            self.m_W[i] = beta1 * self.m_W[i] + (1 - beta1) * dw
            self.v_W[i] = beta2 * self.v_W[i] + (1 - beta2) * (dw ** 2)
            m_hat = self.m_W[i] / (1 - beta1 ** self.t)
            v_hat = self.v_W[i] / (1 - beta2 ** self.t)
            w = w - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            updated_W.append(w)

        for i, (b, db) in enumerate(zip(biases, delta_biases)):
            self.m_B[i] = beta1 * self.m_B[i] + (1 - beta1) * db
            self.v_B[i] = beta2 * self.v_B[i] + (1 - beta2) * (db ** 2)
            m_hat = self.m_B[i] / (1 - beta1 ** self.t)
            v_hat = self.v_B[i] / (1 - beta2 ** self.t)
            b = b - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            updated_B.append(b)

        return updated_W, updated_B

    def train(self, X_train, y_train, X_eval, y_eval, num_epochs, batch_size, optimizer, optimizer_params):
        '''
        Train the neural network.
        
        Parameters
        ----------
        X_train : numpy.ndarray
            Training data of shape (N_train, D).
        y_train : numpy.ndarray
            Training labels of shape (N_train, 1).
        X_eval : numpy.ndarray
            Evaluation data of shape (N_eval, D).
        y_eval : numpy.ndarray
            Evaluation labels of shape (N_eval, 1).
        num_epochs : int
            Number of training epochs.
        batch_size : int
            Size of each training batch.
        optimizer : str
            Optimizer to use ('bgd' for Batch Gradient Descent, 'adam' for Adam optimizer).
        optimizer_params : dict
            Dictionary containing optimizer parameters.
        
        Returns
        -------
        tuple
            Lists of training and testing losses.
        '''
        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            # Shuffle training data
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            X_batches = np.array_split(X_train_shuffled, max(1, X_train_shuffled.shape[0] // batch_size))
            y_batches = np.array_split(y_train_shuffled, max(1, y_train_shuffled.shape[0] // batch_size))

            for X_batch, y_batch in zip(X_batches, y_batches):
                # Forward pass
                self.forward(X_batch)
                dW, db = self.backward(X_batch, y_batch)
                if optimizer == "adam":
                    self.weights, self.biases = self.step_adam(
                        self.weights, self.biases, dW, db, optimizer_params)
                elif optimizer == "bgd":
                    self.weights, self.biases = self.step_bgd(
                        self.weights, self.biases, dW, db, optimizer_params, epoch)
                else:
                    raise ValueError(f"Unsupported optimizer: {optimizer}")

            # Compute the training loss and accuracy
            train_preds = self.forward(X_train)
            # Add epsilon to prevent log(0)
            epsilon = 1e-15
            train_preds = np.clip(train_preds, epsilon, 1 - epsilon)
            train_loss = np.mean(-y_train * np.log(train_preds) - (1 - y_train) * np.log(1 - train_preds))
            train_accuracy = np.mean((train_preds > 0.5).reshape(-1,) == y_train)
            train_losses.append(train_loss)

            # Compute the evaluation loss and accuracy
            test_preds = self.forward(X_eval)
            test_preds = np.clip(test_preds, epsilon, 1 - epsilon)
            test_loss = np.mean(-y_eval * np.log(test_preds) - (1 - y_eval) * np.log(1 - test_preds))
            test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
            test_losses.append(test_loss)

            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        return train_losses, test_losses


    # Plot the loss curve
    def plot_loss(self, train_losses, test_losses, optimizer, optimizer_params):
        '''
        Plot the training and testing loss curves.
        
        Parameters
        ----------
        train_losses : list of float
            Training loss values over epochs.
        test_losses : list of float
            Testing loss values over epochs.
        optimizer : str
            Optimizer used ('bgd' or 'adam').
        optimizer_params : dict
            Dictionary containing optimizer parameters.
        '''
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve using {optimizer.upper()} Optimizer')
        plt.legend()
        if optimizer == "bgd":
            plt.savefig(f'loss_bgd_flag{optimizer_params["gd_flag"]}.png')
        else:
            plt.savefig(f'loss_adam.png')
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Read from data.csv 
    csv_file_path = "data_train.csv"
    eval_file_path = "data_eval.csv"
    
    try:
        data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=0)
        data_eval = np.genfromtxt(eval_file_path, delimiter=',', skip_header=0)
    except IOError:
        print("Error: Could not read data files. Please ensure 'data_train.csv' and 'data_eval.csv' exist.")
        exit()

    # Separate the data into X (features) and y (target) arrays
    X_train = data[:, :-1]
    y_train = data[:, -1]
    X_eval = data_eval[:, :-1]
    y_eval = data_eval[:, -1]

    # Feature scaling (optional, here we square the features as per your example)
    X_train = X_train ** 2
    X_eval = X_eval ** 2

    # Define network architecture and hyperparameters
    input_dim = X_train.shape[1]
    hidden_dims = [4, 2]  # Two hidden layers with 4 and 2 neurons respectively
    num_epochs = 30
    batch_size = 100
    activations = ['sigmoid', 'sigmoid']  # Activation functions for hidden layers
    # optimizer = "bgd"  # Choose between "bgd" and "adam"
    # optimizer_params = {
    #     'learning_rate': 0.03, #lr1 = 0.034 lr2= 0.093 lr3= 0.03
    #     'gd_flag': 3,          # 1: Vanilla GD, 2: GD with Exponential Decay, 3: Momentum
    #     'momentum': 0.099,      # Used if gd_flag == 3 momentum = 0.099
    #     'decay_constant': 0.09 # Used if gd_flag == 2 decay_const = 0.09
    # }
    
    # For Adam optimizer, use the following parameters
    optimizer = "adam"
    optimizer_params = {
        'learning_rate': 0.00087,
        'beta1' : 0.4,
        'beta2' : 0.7,
        'eps' : 1e-8
    }

    # Initialize and train the neural network
    model = NN(input_dim, hidden_dims, activations=activations)
    train_losses, test_losses = model.train(
        X_train, y_train, X_eval, y_eval,
        num_epochs, batch_size, optimizer, optimizer_params
    )
    
    # Final evaluation
    test_preds = model.forward(X_eval)
    test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
    print(f"Final Test accuracy: {test_accuracy:.4f}")

    # Plot the loss curves
    model.plot_loss(train_losses, test_losses, optimizer, optimizer_params)

"""
Activation functions implemented in NumPy

Each activation function includes both the forward pass and derivative
for use in backpropagation.
"""

import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function.
    
    Args:
        x: Input array
        
    Returns:
        Sigmoid(x) = 1 / (1 + e^(-x))
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    """
    Derivative of sigmoid function.
    
    Args:
        x: Input to sigmoid
        
    Returns:
        d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    """
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    """
    Tanh activation function.
    
    Args:
        x: Input array
        
    Returns:
        tanh(x)
    """
    return np.tanh(x)


def tanh_derivative(x):
    """
    Derivative of tanh function.
    
    Args:
        x: Input to tanh
        
    Returns:
        d/dx tanh(x) = 1 - tanh^2(x)
    """
    t = np.tanh(x)
    return 1 - t ** 2


def relu(x):
    """
    ReLU (Rectified Linear Unit) activation function.
    
    Args:
        x: Input array
        
    Returns:
        max(0, x)
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    Derivative of ReLU function.
    
    Args:
        x: Input to ReLU
        
    Returns:
        1 if x > 0, else 0
    """
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function.
    
    Args:
        x: Input array
        alpha: Negative slope (default: 0.01)
        
    Returns:
        x if x > 0, else alpha * x
    """
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    """
    Derivative of Leaky ReLU function.
    
    Args:
        x: Input to Leaky ReLU
        alpha: Negative slope (default: 0.01)
        
    Returns:
        1 if x > 0, else alpha
    """
    return np.where(x > 0, 1.0, alpha)


def softmax(x):
    """
    Softmax activation function for multi-class classification.
    
    Args:
        x: Input array of shape (num_classes, num_samples)
        
    Returns:
        Probabilities that sum to 1 across classes
    """
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)


def softmax_derivative(x):
    """
    Derivative of softmax (Jacobian).
    
    For a single sample, the Jacobian is:
    J_ij = softmax_i * (delta_ij - softmax_j)
    
    Args:
        x: Output of softmax
        
    Returns:
        Jacobian matrix
    """
    s = x.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

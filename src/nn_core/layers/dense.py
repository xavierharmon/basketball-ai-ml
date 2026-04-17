"""
Dense (Fully Connected) Layer implementation
"""

import numpy as np
from ..activations import relu, relu_derivative, sigmoid, sigmoid_derivative


class DenseLayer:
    """
    Fully connected dense layer with configurable activation function.
    
    Performs the operation: output = activation(input @ weights + bias)
    """
    
    def __init__(self, input_size, output_size, activation='relu', learning_rate=0.01):
        """
        Initialize the dense layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features (neurons)
            activation: Activation function ('relu', 'sigmoid', 'linear')
            learning_rate: Learning rate for weight updates
        """
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_name = activation
        
        # Initialize weights with small random values
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((output_size, 1))
        
        # Set activation function
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'linear':
            self.activation = lambda x: x
            self.activation_derivative = lambda x: np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Cache for backpropagation
        self.input = None
        self.z_values = None
        self.output = None
    
    def forward(self, X):
        """
        Forward pass through the layer.
        
        Args:
            X: Input of shape (input_size, batch_size)
            
        Returns:
            Output of shape (output_size, batch_size)
        """
        self.input = X
        self.z_values = np.dot(self.weights, X) + self.bias
        self.output = self.activation(self.z_values)
        return self.output
    
    def backward(self, delta):
        """
        Backward pass (backpropagation) through the layer.
        
        Args:
            delta: Gradient from next layer (output_size, batch_size)
            
        Returns:
            Gradient for previous layer (input_size, batch_size)
        """
        # Apply activation derivative
        delta = delta * self.activation_derivative(self.z_values)
        
        # Compute gradients
        batch_size = self.input.shape[1]
        dW = np.dot(delta, self.input.T) / batch_size
        db = np.sum(delta, axis=1, keepdims=True) / batch_size
        
        # Propagate gradient to previous layer
        input_delta = np.dot(self.weights.T, delta)
        
        # Update weights and bias
        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * db
        
        return input_delta
    
    def get_config(self):
        """Return layer configuration as a dictionary."""
        return {
            'type': 'Dense',
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation': self.activation_name,
            'weights_shape': self.weights.shape,
            'bias_shape': self.bias.shape
        }
    
    def __repr__(self):
        return (f"DenseLayer(input_size={self.input_size}, "
                f"output_size={self.output_size}, "
                f"activation='{self.activation_name}')")

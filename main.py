import numpy as np

# Initialize weights with random values
input_size = 3  # input size
hidden_size = 4  # hidden layer size
output_size = 1  # output size

# Randomly initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward(X):
    z1 = np.dot(X, W1) + b1  # Linear transformation for hidden layer
    a1 = sigmoid(z1)  # Activation function for hidden layer
    
    z2 = np.dot(a1, W2) + b2  # Linear transformation for output layer
    a2 = sigmoid(z2)  # Activation function for output layer
    return a1, a2  # Return both the hidden layer output and final output

def backward(X, Y, a1, output, learning_rate=0.1):
    global W1, W2, b1, b2  # Declare global variables so they can be modified
    
    # Calculate the error (difference between predicted and actual output)
    output_error = Y - output
    output_delta = output_error * sigmoid_derivative(output)
    
    # Backpropagate the error to the hidden layer
    hidden_error = output_delta.dot(W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(a1)
    
    # Update weights and biases using gradient descent
    W2 += a1.T.dot(output_delta) * learning_rate
    b2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(hidden_delta) * learning_rate
    b1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

# Example data
X = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])  # Input data
Y = np.array([[0], [1], [1], [0]])  # Expected output (XOR problem)

# Training loop
for epoch in range(10000):
    a1, output = forward(X)
    backward(X, Y, a1, output)

    if epoch % 1000 == 0:
        loss = np.mean(np.square(Y - output))  # Mean Squared Error
        print(f"Epoch {epoch}, Loss: {loss}")

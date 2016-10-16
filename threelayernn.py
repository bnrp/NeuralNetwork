# Follows github.com/dennybritz/nn-from-scratch
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
matplotlib.use('Agg') # Workaround for tk error
import matplotlib.pyplot as plt



def generate_data():
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    return X, y



def visualize(X, y, model):
    plot_decision_boundary(lambda x: predict(model, x), X , y)
    plt.title("Decision Boundary for Hidden Layer Size 3")



def plot_decision_boundary(pred_func, X, y):
    # Padding
    x_min = X[:, 0].min() - .5
    x_max = X[:, 0].max() + .5
    y_min = X[:, 1].min() - .5
    y_max = X[:, 1].max() + .5
    h = 0.01
    # Generate grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict fn value
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot contour and training set
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.savefig("plot.png")



# Fn to evaluate total dataset loss
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation for preditctions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Loss calculation
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)
    # Add regularization term to loss
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss



# Fn to predict an output
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)



# Learns parameters and returns model
def build_model(nn_hdim, num_passes, print_loss):
    # Initialize the parameters to random vals, need to learn
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
    # Returns this
    model = {}
    # Gradient descent
    for i in range(0, num_passes):
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        # Add regularization terms
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        # Assign new parameters to model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        # Print loss, optionally
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" %(i, calculate_loss(model)))
    return model



def main():
    X, y = generate_data()
    model = build_model(3, 20000, True)
    visualize(X, y, model)



if __name__ == "__main__":
    X, y = generate_data()
    num_examples = len(X) # Training set size
    nn_input_dim = 2
    nn_output_dim =2
    # Gradient Descent
    epsilon = 0.01 # learning rate
    reg_lambda = 0.01 # regularization strength
    main()

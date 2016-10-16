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



def classify(X, y):
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X, y)
    return clf



def visualize(X, y, clf):
    plot_decision_boundary(lambda x: clf.predict(x), X, y)
    plt.title("Logistic Regression")



def plot_decision_boundary(pred_func, X, y):
    # Padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
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



def main():
    X, y = generate_data()
    clf = classify(X, y)
    visualize(X, y, clf)



if __name__ == "__main__":
    main()

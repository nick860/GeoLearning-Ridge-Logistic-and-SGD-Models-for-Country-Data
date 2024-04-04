# GeoLearning-Ridge-Logistic-and-SGD-Models-for-Country-Data

Welcome to ML Algorithms Demos in Python! This repository contains Python scripts demonstrating the implementation and usage of various machine learning algorithms. From ridge regression to decision trees, you'll find practical examples and explanations to understand how these algorithms work and how to use them in your projects.

## Overview

The repository consists of the following files:

- `ridge_regression.py`: Implementation of Ridge Regression for linear regression tasks.
- `logistic_regression.py`: Implementation of Logistic Regression for binary classification tasks.
- `stochastic_gradient_descent.py`: Implementation of Stochastic Gradient Descent for training logistic regression models with PyTorch.
- `decision_trees.py`: Implementation of Decision Trees for classification tasks using scikit-learn.
- `helpers.py`: Helper functions for data preprocessing, visualization, and model evaluation.
- `README.md`: This file, providing an overview of the repository and usage instructions.

## Files Description

### ridge_regression.py

This file contains the implementation of the `Ridge_Regression` class for performing ridge regression. Key functionalities include:

- **Ridge_Regression Class**: Implements ridge regression using the closed-form solution or matrix inversion method. The `fit` method fits the model to training data, while the `predict` method predicts outputs for new data.

### logistic_regression.py

Here, you'll find the `Logistic_Regression` class for logistic regression tasks. This class is implemented using PyTorch and includes methods for training the model (`forward`) and making predictions (`predict`).

### stochastic_gradient_descent.py

The `Stochastic_gradient_descent` function in this file demonstrates the usage of Stochastic Gradient Descent (SGD) for training logistic regression models. It includes options for regularization, learning rate decay, and handling multi-class classification.

### decision_trees.py

This file provides functions for training and visualizing decision tree classifiers using scikit-learn's `DecisionTreeClassifier`. Key functionalities include training decision trees with different depths and visualizing decision boundaries.

### helpers.py

The `helpers.py` file contains various helper functions used across different scripts, such as data reading, plotting decision boundaries, and evaluating model performance.

## Usage

To utilize the functionalities provided in these files, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies: NumPy, pandas, scikit-learn, PyTorch, and matplotlib.
3. Execute the desired Python script (e.g., `python ridge_regression.py`) to run the algorithm and observe the results.

## Dependencies

- NumPy
- pandas
- scikit-learn
- PyTorch
- matplotlib

Feel free to explore and experiment with the code provided in this repository!

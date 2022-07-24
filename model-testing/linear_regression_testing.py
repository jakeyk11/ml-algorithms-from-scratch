# %% Testing linear regression model

# %% Imports

import os
import sys
import numpy as np

# %% Add custom tools to path

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
from ml_algorithms.linear_regression import SimpleLinearRegression

# %% Create test data

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# %% Fit model and make predictions

lin_model = SimpleLinearRegression(visualisation=True)
lin_model.fit(xs, ys)

x_predict = np.array([-2, 8, 3.6])
y_predict = lin_model.predict(x_predict)

# %% Fit model with forced zero intercept

lin_model_zero = SimpleLinearRegression(visualisation=True, zero_int=True)
lin_model_zero.fit(xs, ys)

y_predict_z = lin_model_zero.predict(x_predict)

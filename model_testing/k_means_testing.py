# %% Testing linear regression model

# %% Imports

import os
import sys
import numpy as np

# %% Add custom tools to path

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
from ml_algorithms.k_means import KMeans

# %% Create test data

X = np.array([[1,2], [1.5,1.8], [5,8], [8,8], [1,0.6], [9,11], [7,2],
              [2,4], [4,7], [6,2], [7,4], [8,1], [6,10], [6,6]])

# %% Fit model and make predictions

km_model = KMeans(k=3, visualisation=True)
km_model.fit(X)

new_features = np.array([[1,3], [8,9],[0,3],[5,4]])
class_predict = km_model.predict(new_features)

# %% Visualise model centroids

km_model.visualise()
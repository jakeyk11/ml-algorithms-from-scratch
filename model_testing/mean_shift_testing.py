# %% Testing linear regression model

# %% Imports

import os
import sys
import numpy as np
import random
from sklearn.datasets import make_blobs

# %% Add custom tools to path

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
from ml_algorithms.mean_shift import MeanShift

# %% Create test data

centers = random.randrange(2,5)
X, y = make_blobs(n_samples=50, centers=centers, n_features=2)

# %% Fit model and make predictions

ms_model = MeanShift(visualisation=True)
ms_model.fit(X)

new_features = np.array([[1,3], [8,9],[0,3],[5,4]])
class_predict = ms_model.predict(new_features)

# %% Visualise model centroids

ms_model.visualise()

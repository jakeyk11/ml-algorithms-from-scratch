# %% Testing linear regression model

# %% Imports

import os
import sys

# %% Add custom tools to path

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
from ml_algorithms.support_vector_machine import SupportVectorMachine

# %% Create test data

data = {'red':[[2,3],[3,1],[5,3],[3,6],[8,1]], 'blue':[[6,5],[2,8.5],[8,6],[5,8],[7,10]]}

# %% Fit model and make predictions

svm_model = SupportVectorMachine(visualisation=True)
svm_model.fit(data)

new_features = [[4,5.5],[4,7],[5,2],[7,4]]
class_predict = svm_model.predict(new_features)

# %% Visualise model seperating hyperplane 

svm_model.visualise()


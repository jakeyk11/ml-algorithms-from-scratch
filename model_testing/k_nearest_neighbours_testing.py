# %% Testing k nearest neighbours model

# %% Imports

import os
import sys
import pandas as pd
import random

# %% Add custom tools to path

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
from ml_algorithms.k_nearest_neighbours import KNearestNeighbours

# %% Create test data

data = {'red':[[1,2],[2,3],[3,1]], 'black':[[6,5],[7,7],[8,6]], 'blue':[[1,8], [3,6], [3,7]], 'green':[[8,1], [8,2], [5,1]]}

# %% Fit model and make predictions

knn_model = KNearestNeighbours(visualisation=True, k=3)
knn_model.fit(data)

new_features = [[5,7],[7,5],[2,2],[4,7]]
y_predict, y_confidence = knn_model.predict(new_features)

# %% Visualise model decision boundaries

knn_model.visualise(h=0.5)

# %% With a working model, apply it to higher dimensional problem - breast cancer wisconsin dataset

# Import dataset, replace missing data and remove useless fields
df = pd.read_csv("../datasets/breast-cancer-wisconsin.data")
df.replace('?',-99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)

# Convert to list, ensuring all values are floats
full_data = df.astype(float).values.tolist()

# Manual implementation of test, train split.
random.shuffle(full_data)
test_size = 0.2
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

# Create train class dictionary for use in knn function (2 = benign, 4 = malignant)
train_dict = {2:[], 4:[]}

for i in train_data:	
	train_dict[i[-1]].append(i[:-1])

test_features = [item[:-1] for item in test_data]
test_classes = [item[-1] for item in test_data]
								
# Fit model on training data
knn_model = KNearestNeighbours(visualisation=False, k=3)
knn_model.fit(train_dict)

# Make predictions on test data
y_predict, y_confidence = knn_model.predict(test_features)

# Assess basic model accuracy
correct = 0
for i in range(0, len(y_predict)):
    if y_predict[i] == test_classes[i]:
        correct += 1
test_accuracy = correct/len(y_predict)                                          # 98.5% accuracy

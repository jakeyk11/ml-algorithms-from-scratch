"""Module containing k nearest neighbours models for implementation within data science projects.

Classes
---------
kNearestNeighbours
    k nearest neighbours classification model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import warnings
from collections import Counter


class KNearestNeighbours:
    """k nearest neighbours classification model.

    k nearerst neighbours supervised machine learning model for classification. Compares the Euclidean distance
    between the feature and neighbouring features to determine which class it belongs to.

    Attributes:
        data (dict): nested list of features (values) that correspond to each class (keys)
        class_count (int): number of classes.
        class_names (list): class names.
        colours (tuple): colours to be associated with classes
        class_colours (dict): mapping of classes to colours
        visualisation (bool): Specification of whether predicted response is graphically displayed.
        fig (matplotlib.pyplot.figure): Figure window on which model is displayed (None if not displayed)
        ax (matplotlib.pyplot.axes): Axes object on which model is displayed (None if not displayed)
        """

    def __init__(self, k=3, visualisation=True, colours=('r', 'k', 'b', 'g', 'c', 'y', 'm')):
        """Initialises kNearestNeighbours class.

        Args:
            k (int, optional): Number of nearest neighbours to use for classification. 3 by default.
            visualisation (boolean, optional): toggle visualisation of model. True by default.
            colours (tuple, optional): colours to be associated with classes. ('r', 'k', 'b', 'g', 'c', 'y', 'm') by default.
            """

        # Assign attributes
        self.k = k
        self.visualisation = visualisation
        self.data = None
        self.class_count = None
        self.class_names = None
        self.colours = colours
        self.class_colours = None

        # Create figure attributes, based on visualisation toggle
        if self.visualisation:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.fig.set_facecolor('#313332')
            self.ax.patch.set_alpha(0)
            self.ax.grid(visible=True, alpha=0.1)
            mpl.rcParams['xtick.color'] = 'w'
            mpl.rcParams['ytick.color'] = 'w'
            mpl.rcParams['xtick.labelsize'] = 10
            mpl.rcParams['ytick.labelsize'] = 10
            spines = ["top", "right", "bottom", "left"]
            for s in spines:  # Remove top and right spine, and change colour of botton and left to white
                if s in ["top", "right"]:
                    self.ax.spines[s].set_visible(False)
                else:
                    self.ax.spines[s].set_color('w')

        else:
            self.fig = None
            self.ax = None

    def fit(self, data):
        """Determine number of classes, and names of classes in k nearest neighbours classification model.

        Args:
            data (dict): nested list of features (values) that correspond to each class (keys)"""

        # Overwrite data attributes
        self.data = data

        # Warn if number of nearest neighbours is larger than number of available features
        if sum(len(v) for v in data.values()) <= self.k:
            warnings.warn('k is set to a value less than total available features!')

        # Number of classes and class names
        self.class_count = len(data.keys())
        self.class_names = list(data.keys())

        # Show first two dimensions of data
        if self.visualisation:
            self.class_colours = dict()
            for i, class_to_plot in enumerate(self.data):
                for feature_to_plot in self.data[class_to_plot]:
                    self.ax.scatter(feature_to_plot[0], feature_to_plot[1], s=50, color=self.colours[i])
                self.class_colours[class_to_plot] = self.colours[i]
            plt.title(f"k Nearest Neighbours Model: k={self.k}, {self.class_count} classes", fontweight="bold", color="w")

    def predict(self, features_predict):
        """Make predictions with k nearest neighbours model.

        Args:
            features_predict (list): List of features to classify

        Returns:
            y_class (list): Class to which each feature belongs
            y_confidence (list): Confidence associated with each class prediction."""

        # Initialise class prediction and confidence lists
        y_class = []
        y_confidence = []

        # Initialise list of distances to each feature
        for feature_predict in features_predict:
            distances = []

            # Calculate distances to all the other features
            for feature_class in self.data:
                for features in self.data[feature_class]:
                    euclid_distance = np.linalg.norm(np.array(features) - np.array(feature_predict))
                    distances.append([euclid_distance, feature_class])

            # Create list of classes corresponding to k nearest features.
            votes = [i[1] for i in sorted(distances)[:self.k]]

            # Determine most common class (=prediction) and confidence.
            current_class = Counter(votes).most_common(1)[0][0]
            y_class.append(current_class)
            y_confidence.append(Counter(votes).most_common(1)[0][1] / self.k)

            # Show first two dimensions of data
            if self.visualisation:
                self.ax.scatter(feature_predict[0], feature_predict[1], s=50, color=self.class_colours[current_class],
                                edgecolor='w')

        return y_class, y_confidence

    def visualise(self, h=0.5):
        """Show k nearest neighbours decision boundaries. Only available for 2D features.

        Args:
            h (float, optional): Step-size for mesh of points used to create decision boundaries. Default is 0.5."""

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.set_facecolor('#313332')
        ax.patch.set_alpha(0)
        ax.grid(visible=True, alpha=0.1)
        mpl.rcParams['xtick.color'] = 'w'
        mpl.rcParams['ytick.color'] = 'w'
        mpl.rcParams['xtick.labelsize'] = 10
        mpl.rcParams['ytick.labelsize'] = 10
        spines = ["top", "right", "bottom", "left"]
        for s in spines:  # Remove top and right spine, and change colour of botton and left to white
            if s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            else:
                ax.spines[s].set_color('w')

        # Define colourmap
        cmap_light = ListedColormap(list(self.colours[0:self.class_count]))

        # Calculate minimum and maximum co-ordinates of training data-set
        xs = [feature[0] for feature in [x for xs in self.data.values() for x in xs]]
        ys = [feature[1] for feature in [y for ys in self.data.values() for y in ys]]
        x_min, x_max = min(xs) - 1, max(xs) + 1
        y_min, y_max = min(ys) - 1, max(ys) + 1

        # Mesh-grid of points
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        self.visualisation = False
        z, z_conf = KNearestNeighbours.predict(self, np.c_[xx.ravel(), yy.ravel()].tolist())

        # Convert classes to numerical identifiers
        z_identifier = z[:]
        for i, class_tag in enumerate(self.class_names):
            for ii, value in enumerate(z):
                if value == class_tag:
                    z_identifier[ii] = i

        # Put the result into a color plot
        z_identifier = np.array(z_identifier).reshape(xx.shape)

        ax.pcolormesh(xx, yy, z_identifier, shading='auto', cmap=cmap_light, alpha=0.2)

        # Plot also the training points
        for i, point in enumerate(np.c_[xx.ravel(), yy.ravel()].tolist()):
            ax.scatter(point[0], point[1], c=self.class_colours[z[i]], s=z_conf[i]*h*100)

        plt.title(f"kNN (k={self.k}) decision boundaries and prediction mesh.\nDot size indicates confidence",
                  fontweight="bold", color="w")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

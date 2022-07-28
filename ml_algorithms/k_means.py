"""Module containing k means models for implementation within data science projects.

Classes
---------
kMeans
    k means multidimensional flat clustering model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings


class KMeans:
    """k means multidimensional flat clustering model.

    k means unsupervised machine learning model for classification. Based on a definition of the number of clusters,
    centroids are randomly positioned and features classified based on the centroid to which they are closest.
    Centroids are re-positioned at the mean location of the new "feature sets". This process is repeated until the
    centroids move within a defined tolerance (or maximum number of steps).

    Attributes:
        data (numpy.array): array of features (values) to classify.
        k (int): number of classes or clusters.
        tol (float): tolerance, or percentage distance below which centroids must move to complete clustering.
        max_iter (int): maximum number of iterations permissible.
        centroids (dict): position of centroid for each class
        classifications (dict): data points classified against each class
        colours (tuple): colours to be associated with clusters
        class_colours (dict): mapping of classes to colours
        visualisation (bool): Specification of whether data and predictions are graphically displayed.
        fig (matplotlib.pyplot.figure): Figure window on which model is displayed (None if not displayed)
        ax (matplotlib.pyplot.axes): Axes object on which model is displayed (None if not displayed)
        """

    def __init__(self, k=2, tol=0.001, max_iter=300, visualisation=True, colours=('r', 'k', 'b', 'g', 'c', 'y', 'm')):
        """Initialises KNearestNeighbours class.

        Args:
            k (int, optional): Number of classes or clusters. 2 by default.
            tol (float, optional): tolerance, or percentage distance below which centroids must move to complete clustering. 0.001 by default.
            max_iter (int, optional): maximum number of iterations permissible. 300 by default.
            visualisation (boolean, optional): toggle visualisation of model. True by default.
            colours (tuple, optional): colours to be associated with classes. ('r', 'k', 'b', 'g', 'c', 'y', 'm') by default.
            """

        # Assign attributes
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.visualisation = visualisation
        self.class_colours = None
        self.data = None
        self.colours = colours
        self.centroids = {}
        self.classifications = {}

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
        """Fit k Means model

        Args:
            data (numpy.array): array of features (values) to classify."""

        # Overwrite data attribute
        self.data = data

        # Warn if number of classes is larger than number of available features
        if len(data) <= self.k:
            warnings.warn('k is set to a value greater than total available features!')

        # Set the initial location of the k centroids to be at the first k data points
        for i in range(self.k):
            self.centroids[i] = data[i]

        # Begin optimisation process, within constraint of maximum iterations
        for iteration in range(self.max_iter):
            self.classifications = {}

            # Initialise classification dictionary for each iteration
            for i in range(self.k):
                self.classifications[i] = []

            # For each feature, calculate its distance to each centroid and classify based on smallest distance
            for feature in data:
                distances = [np.linalg.norm(feature - self.centroids[c]) for c in self.centroids]
                closest_centroid = distances.index(min(distances))

                # Build up clusters
                self.classifications[closest_centroid].append(feature)

            # Before repositioning centroids, store current value for later comparison
            previous_centroids = dict(self.centroids)

            # Reposition centroids based on mean of each cluster
            for cluster in self.classifications:
                self.centroids[cluster] = np.average(self.classifications[cluster], axis=0)

            optimised = True

            # Check whether centroid movement meets tolerance criteria
            for c in self.centroids:
                previous_centroid = previous_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - previous_centroid) / previous_centroid) > self.tol:
                    optimised = False

            # Break loop if optimisation is complete
            if optimised:
                break

        # Show first two dimensions of data
        if self.visualisation:
            self.class_colours = dict()
            for i, cluster in enumerate(self.classifications):
                for feature in self.classifications[cluster]:
                    self.ax.scatter(feature[0], feature[1], s=50, color=self.colours[i])
                self.class_colours[cluster] = self.colours[i]
            plt.title(f"k Means Model: {self.k} classes", fontweight="bold", color="w")

    def predict(self, features_predict):
        """Make predictions with k means model.

        Args:
            features_predict (list): List of features to classify

        Returns:
            y_class (list): Class to which each feature belongs."""

        # Initialise class prediction list
        y_class = []

        # For each feature, determine the closest centroid and classify
        for feature_predict in features_predict:
            distances = [np.linalg.norm(feature_predict - self.centroids[c]) for c in self.centroids]
            current_class = distances.index(min(distances))
            y_class.append(current_class)

            # Show first two dimensions of data
            if self.visualisation:
                self.ax.scatter(feature_predict[0], feature_predict[1], s=50, color=self.class_colours[current_class],
                                edgecolor='w')

        return y_class

    def visualise(self):
        """Show k means centroids. Only available for 2D features."""

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
        plt.title(f"k Means (k={self.k}) centroids", fontweight="bold", color="w")

        # Plot training points
        for i, cluster in enumerate(self.classifications):
            for feature in self.classifications[cluster]:
                ax.scatter(feature[0], feature[1], s=50, color=self.colours[i])

        # Plot centroids
        for c in self.centroids:
            ax.scatter(self.centroids[c][0], self.centroids[c][1], color="w", s=50)
            ax.scatter(self.centroids[c][0], self.centroids[c][1], color="w", facecolors='none', s=150)

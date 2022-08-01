"""Module containing mean shift models for implementation within data science projects.

Classes
---------
MeanShift
    Mean shift multidimensional hierarchical clustering model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class MeanShift:
    """Mean shift multidimensional flat clustering model.

    Mean shift unsupervised machine learning model for classification. Every feature is initially defined as a
    cluster centre, with a bandwidth (area) created around it. Other features within the bandwidth are identified,
    and the mean used to define new cluster centers. This process is repeated until the centroids move within a
    defined tolerance (or maximum number of steps). With a simple bandwidth definition, clusters quickly capture too
    many data points and a hierarchy of bandwidths, or dynamic bandwidth, is applied to help with this.

    Attributes:
        data (numpy.array): array of features (values) to classify.
        bandwidth (float): radius for grouping centroids.
        bandwidth_norm_step (float): step size for application of dynamic bandwidth.
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

    def __init__(self, bandwidth=None, bandwidth_norm_step=100, tol=0.0001, max_iter=1000000, visualisation=True,
                 colours=('r', 'k', 'b', 'g', 'c', 'y', 'm')):
        """Initialises MeanShift class.

        Args:
            bandwidth (float, optional): radius for grouping centroids. None by default, triggering automatic calculation.
            bandwidth_norm_step (float, optional): step size for application of dynamic bandwidth. 100 by default.
            tol (float, optional): tolerance, or percentage distance below which centroids must move to complete clustering. 0.0001 by default.
            max_iter (int, optional): maximum number of iterations permissible. 1000000 by default.
            visualisation (boolean, optional): toggle visualisation of model. True by default.
            colours (tuple, optional): colours to be associated with classes. ('r', 'k', 'b', 'g', 'c', 'y', 'm') by default.
            """

        # Assign attributes
        self.bandwidth = bandwidth
        self.bandwidth_norm_step = bandwidth_norm_step
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

        # Calculate appropriate bandwidth if user does not specify (by defining magnitude of centroid on all data)
        if self.bandwidth is None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.bandwidth = all_data_norm / self.bandwidth_norm_step

        # Set the initial location of centroids to be at the data point positions
        for i in range(len(data)):
            self.centroids[i] = data[i]

        # Define weights as a list starting at bandwidth_norm_step-1, and reducing down to 1.
        weights = [i for i in range(self.bandwidth_norm_step)][::-1]

        # Begin optimisation process, within constraint of maximum iterations
        for iteration in range(self.max_iter):
            new_centroids = []

            # For each centroid, determine data points within its bandwidth and hence updated centroid position
            for i in self.centroids:
                in_bandwidth = []
                centroid = self.centroids[i]

                # For each feature, calculate its distance to centroid and check if within bandwidth
                for feature in data:
                    distance = np.linalg.norm(feature - centroid)

                    # If distance is 0 (as in 1st step where centroids = features), adjust for mathematical convenience
                    if distance == 0:
                        distance = 0.000001

                    # Determine weight index for features within bandwidth, based on proximity to centroid
                    weight_index = int(distance / self.bandwidth)
                    if weight_index > self.bandwidth_norm_step - 1:
                        weight_index = self.bandwidth_norm_step - 1

                    # Apply weighting through duplicating features according to weight^2
                    features_to_add = (weights[weight_index] ** 2) * [feature]
                    in_bandwidth += features_to_add

                # Update centroid position and add to list of centroids
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            # Define sorted list of unique centroids in current iteration
            unique_centroids = sorted(list(set(new_centroids)))
            to_pop = []

            # Remove centroids that are within one bandwidth of one another, by tagging them in the to_pop list
            for unique_centroid in unique_centroids:
                if unique_centroid in to_pop:
                    pass
                for another_unique_centroid in unique_centroids:
                    if unique_centroid == another_unique_centroid:
                        pass
                    elif (np.linalg.norm(np.array(unique_centroid) - np.array(another_unique_centroid)) <=
                          self.bandwidth) and another_unique_centroid not in to_pop:
                        to_pop.append(another_unique_centroid)
            for centroid_to_remove in to_pop:
                unique_centroids.remove(centroid_to_remove)

            # Before repositioning centroids, store current value for later comparison
            previous_centroids = dict(self.centroids)

            # Reposition centroids
            self.centroids = {}
            for centroid in range(len(unique_centroids)):
                self.centroids[centroid] = np.array(unique_centroids[centroid])

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

        # Initialise classifications list with cluster keys
        for cluster in range(len(self.centroids)):
            self.classifications[cluster] = []

        # Assign each feature to a cluster based on distance to centroids
        for feature in data:
            distances = [np.linalg.norm(feature - self.centroids[c]) for c in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(feature)

        # Show first two dimensions of data
        if self.visualisation:
            self.class_colours = dict()
            for i, cluster in enumerate(self.classifications):
                for feature in self.classifications[cluster]:
                    self.ax.scatter(feature[0], feature[1], s=50, color=self.colours[i])
                self.class_colours[cluster] = self.colours[i]
            plt.title(f"Mean Shift Model: {len(self.classifications)} classes, Bandwidth = {round(self.bandwidth,2)}",
                      fontweight="bold", color="w")

    def predict(self, features_predict):
        """Make predictions with mean shift model.

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
        """Show mean shift centroids. Only available for 2D features."""

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
        plt.title(f"Mean Shift Centroids", fontweight="bold", color="w")

        # Plot training points
        for i, cluster in enumerate(self.classifications):
            for feature in self.classifications[cluster]:
                ax.scatter(feature[0], feature[1], s=50, color=self.colours[i])

        # Plot centroids
        for c in self.centroids:
            ax.scatter(self.centroids[c][0], self.centroids[c][1], color="w", s=50)
            ax.scatter(self.centroids[c][0], self.centroids[c][1], color="w", facecolors='none', s=150)

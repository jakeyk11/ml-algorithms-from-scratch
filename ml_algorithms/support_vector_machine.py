"""Module containing support vector machine models for implementation within data science projects.

Classes
---------
SupportVectorMachine
    Support vector machine 2D binary classification model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class SupportVectorMachine:
    """Support vector machine 2D binary classification model.

    Support vector machine supervised machine learning model for binary classification. Generates the hyper-plane
    that best seperates two classes that are represented by 2D data.

    Attributes:
        data (dict): nested list of features (values) that correspond to each class (keys)
        class_count (int): number of classes.
        class_names (list): class names.
        colours (tuple): colours to be associated with each class
        class_colours (dict): mapping of classes to colours
        max_feature_value (float): maximum value of any feature in the training data set.
        min_feature_value (float): minimum value of any feature in the training data set.
        w (numpy.array): description of vector through origin perpendicular to the svm seperating hyperplane
        b (float): constant or bias that dictates translation/position of seperating hyperplane
        visualisation (bool): Specification of whether predicted response is graphically displayed.
        fig (matplotlib.pyplot.figure): Figure window on which model is displayed (None if not displayed)
        ax (matplotlib.pyplot.axes): Axes object on which model is displayed (None if not displayed)
        """

    def __init__(self, visualisation=True, colours=('r', 'b')):
        """Initialises SupportVectorMachine class.

        Args:
            visualisation (boolean, optional): toggle visualisation of model. True by default.
            colours (tuple, optional): colours to be associated with classes. ('r', 'b') by default.
            """

        # Assign attributes
        self.visualisation = visualisation
        self.data = None
        self.class_count = None
        self.class_names = None
        self.colours = colours
        self.class_colours = None
        self.max_feature_value = None
        self.min_feature_value = None
        self.w = None
        self.b = None

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
        """Fit Support Vector Machine

        Args:
            data (dict): nested list of features (values) that correspond to each class (keys)"""

        # Overwrite data attribute
        self.data = data

        # Number of classes and class names
        self.class_count = len(data.keys())
        self.class_names = list(data.keys())

        # Create y_i dictionary, assigning classes to y_i =-1 and y_i=+1
        yi_dict = {-1: list(self.data.values())[0], 1: list(self.data.values())[1]}

        # Determine the minimum and maximum feature values (x or y)
        all_data = []
        for select_class in self.data:
            for feature_set in self.data[select_class]:
                for feature in feature_set:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        # Use max feature value to determine various step sizes for optimising the vector W and constant b
        step_sizes = [self.max_feature_value * 0.25,         # Start with step 10% of largest feature
                      self.max_feature_value * 0.1,        # Reduce step size as we get closer to optimal W and b
                      self.max_feature_value * 0.01]       # Continue step size reduction to point of expense
        b_range_multiple = 5
        b_step_size_multiple = 5

        # Use max feature value to produce an initial guess of each component of W (corner cut here)
        latest_optimum = self.max_feature_value * 10

        # Initialise optimisation dictionary, capturing ||W||'s as keys and [w,b]'s as values
        optimisation_dict = {}

        # Search for the vector W and constant b that satisfy y_i(x_i·W+b)>=1 for all training data
        for step in step_sizes:

            # To speed things up, assume W components are equal and the vector W is therefore at 45deg
            w = np.array([latest_optimum, latest_optimum])
            w0_optimised = False
            w1_optimised = False

            # Whilst still searching for the second component of W
            while not w1_optimised:

                # Whilst still searching for the first component of W
                while not w0_optimised:

                    # Consider b values within range of W, where range is defined by b_range_multiple
                    for b in np.arange(-1*self.max_feature_value*b_range_multiple,
                                       self.max_feature_value*b_range_multiple, step*b_step_size_multiple):

                        # Consider each direction of the vector W, and start by assuming we will find a valid W, b
                        for transformation in [[1, 1], [-1, 1], [-1, -1], [1, -1]]:
                            w_transformed = w * transformation
                            found_option = True

                            # For each feature_set (which corresponds to a class and therefore a y_i value)
                            for yi in yi_dict:
                                for feature_set in yi_dict[yi]:

                                    # Check for the correct W, b combination: y_i(x_i·W + b) >= 1
                                    if not yi*(np.dot(w_transformed, feature_set) + b) >= 1:
                                        found_option = False

                            # If found_option has not been reset to false, we have a valid W, b for the training data.
                            if found_option:
                                optimisation_dict[np.linalg.norm(w_transformed)] = [w_transformed, b]

                    # Step through 1st component of W, unless all W[0] at current step size have been checked (w[0] < 0)
                    if w[0] < 0:
                        w0_optimised = True
                    else:
                        w[0] = w[0] - step

                # Step through 2nd component of W, unless all W[1] at current step size have been checked (w[1] < 0)
                if w[1] < 0:
                    w1_optimised = True
                else:
                    w[1] = w[1] - step

                    # Reset 1st component of 1 when stepping through 2nd component of W.
                    w[0] = latest_optimum
                    w0_optimised = False

            # To maximise hyperplane street width we minimise ||W||, so pick W based on smallest ||W||
            w_magnitudes_sorted = sorted([norm for norm in optimisation_dict])
            optimised_choice = optimisation_dict[w_magnitudes_sorted[0]]
            self.w = optimised_choice[0]
            self.b = optimised_choice[1]

            # For the following (smaller) step size, define the latest optimum W as he first element of W + twice step
            latest_optimum = optimised_choice[0][0] + step*2

        # Show first two dimensions of data
        if self.visualisation:
            self.class_colours = dict()
            for i, class_to_plot in enumerate(self.data):
                for feature_to_plot in self.data[class_to_plot]:
                    self.ax.scatter(feature_to_plot[0], feature_to_plot[1], s=50, color=self.colours[i])
                self.class_colours[class_to_plot] = self.colours[i]
            plt.title(f"Support Vector Machine, {self.class_count} classes", fontweight="bold", color="w")

    def predict(self, features_predict):
        """Make predictions with Support Vector Machine

        Args:
            features_predict (list): List of features to classify

        Returns:
            y_class (list): Class to which each feature belongs"""

        # Initialise class prediction list
        y_class = []

        # Determine the sign of x_i·W+b, for each feature x_i.
        for feature in features_predict:
            svm_class = np.sign(np.dot(np.array(feature), self.w) + self.b)

            # Relate the svm class back to the class names
            if svm_class == -1:
                current_class = self.class_names[0]
            elif svm_class == 1:
                current_class = self.class_names[1]
            else:
                current_class = svm_class

            # Add current class to prediction list
            y_class.append(current_class)

            # Show classified feature
            if current_class != 0 and self.visualisation:
                self.ax.scatter(feature[0], feature[1], s=50, color=self.class_colours[current_class],
                                edgecolor='w')

        return y_class

    def visualise(self):
        """Show Support Vector Machine separating hyper-plane."""

        # Create function to calculate
        def hyperplane(x, w, b, val):
            return (-w[0]*x - b + val)/w[1]

        # Set up plot
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

        if self.b/self.w[1] > 0:
            plt.title(f"SVM separating hyperplane. Decision boundary y = {round(-self.w[0]/self.w[1], 2)}x + {round(self.b / self.w[1], 2)}",
                      fontweight="bold", color="w")

        elif self.b/self.w[1] < 0:
            plt.title(f"SVM separating hyperplane. Decision boundary y = {round(-self.w[0]/self.w[1], 2)}x {round(self.b / self.w[1], 2)}",
                      fontweight="bold", color="w")

        else:
            plt.title(f"SVM separating hyperplane. Decision boundary y = {round(-self.w[0]/self.w[1], 2)}x",
                      fontweight="bold", color="w")

        # Plot training points
        for i, class_to_plot in enumerate(self.data):
            for feature_to_plot in self.data[class_to_plot]:
                ax.scatter(feature_to_plot[0], feature_to_plot[1], s=50, color=self.colours[i], zorder=2)

        # Define hyperplane end points based on extreme data points
        hyperplane_x_min = self.min_feature_value*0.9
        hyperplane_x_max = self.max_feature_value*1.1

        # Create positive support vector points
        psv_y_for_xmin = hyperplane(hyperplane_x_min, self.w, self.b, 1)
        psv_y_for_xmax = hyperplane(hyperplane_x_max, self.w, self.b, 1)

        # Create negative support vector points
        nsv_y_for_xmin = hyperplane(hyperplane_x_min, self.w, self.b, -1)
        nsv_y_for_xmax = hyperplane(hyperplane_x_max, self.w, self.b, -1)

        # Create decision boundary points
        db_y_for_xmin = hyperplane(hyperplane_x_min, self.w, self.b, 0)
        db_y_for_xmax = hyperplane(hyperplane_x_max, self.w, self.b, 0)

        # Plot vectors
        ax.plot([hyperplane_x_min, hyperplane_x_max], [psv_y_for_xmin, psv_y_for_xmax], 'w', zorder=1)
        ax.plot([hyperplane_x_min, hyperplane_x_max], [nsv_y_for_xmin, nsv_y_for_xmax], 'w', zorder=1)
        ax.plot([hyperplane_x_min, hyperplane_x_max], [db_y_for_xmin, db_y_for_xmax], 'y--', zorder=1)

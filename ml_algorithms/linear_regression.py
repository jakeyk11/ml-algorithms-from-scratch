"""Module containing linear regression models for implementation within data science projects.

Classes
---------
SimpleLinearRegression
    Simple linear regression model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class SimpleLinearRegression:
    """Simple linear regression model.

    Linear regression model for describing the relationship between a scalar response and a single explanatory variable.

    Attributes:
        x (numpy.array): Single feature on which the model is fit.
        y (numpy.array): Scalar response on which the model is fit.
        m (float): Gradient of linear model fit.
        c (float): y-intercept of linear model fit.
        zero_int (bool): Specification of whether y-intercept is forced to zero.
        visualisation (bool): Specification of whether predicted response is graphically displayed.
        fig (matplotlib.pyplot.figure): Figure window on which model is displayed (None if not displayed)
        ax (matplotlib.pyplot.axes): Axes object on which model is displayed (None if not displayed)
        """

    def __init__(self, visualisation=True, zero_int=False):
        """Initialises SimpleLinearRegression class visualisation toggle and zero y-intercept toggle.

        Args:
            visualisation (boolean): toggle visualisation of model.
            zero_int (boolean): specify whether to force y-intercept to zero."""

        # Assign attributes
        self.visualisation = visualisation
        self.zero_int = zero_int
        self.x = None
        self.y = None
        self.m = None
        self.c = None

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

    def fit(self, x, y):
        """Fit linear regression model.

        Args:
            x (numpy.array): 1D array of feature values.
            y (numpy.array): 1D array of response values, corresponding to feature values."""

        # Overwrite data attributes
        self.x = x
        self.y = y

        # Calculate linear model gradient and y-intercept using ordinary least squares
        if self.zero_int:
            self.m = np.sum(self.x * self.y) / np.sum(self.x**2)
            self.c = 0
        else:
            self.m = np.sum((self.x - np.mean(self.x)) * (self.y - np.mean(self.y))) / \
                     np.sum((self.x - np.mean(self.x))**2)
            self.c = np.mean(self.y) - self.m * np.mean(self.x)

        # Show model fit
        if self.visualisation:
            self.ax.scatter(self.x, self.y, s=50, marker="o", c='grey')
            self.ax.plot(self.x, self.m * self.x + self.c, c='w')
            if self.c > 0:
                plt.title(f"Simple Linear Regression Model: y = {np.round(self.m,2)}x + {np.round(self.c,2)}",
                          fontweight="bold", color="w")
            elif self.c < 0:
                plt.title(f"Simple Linear Regression Model: y = {np.round(self.m,2)}x {np.round(self.c,2)}",
                          fontweight="bold", color="w")
            else:
                plt.title(f"Simple Linear Regression Model: y = {np.round(self.m,2)}x", fontweight="bold", color="w")

    def predict(self, features):
        """Make predictions with linear regression model.

        Args:
            features (numpy.array): 1D array of feature values to predict on.

        Returns:
            y_predict (numpy.array): Predicted response variable"""

        y_predict = self.m * features + self.c

        # Show prediction
        if self.visualisation:
            self.ax.scatter(features, y_predict, s=50, marker="o", c='r')
            self.ax.plot(np.append(self.x, features), np.append(self.m * self.x + self.c, y_predict), c='w')

        return y_predict

# Machine Learning Algorithms from Scratch
This repository contains a collection of Machine learning algorithms developed from scratch within Python.

## Contents

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#project-description"> ➤ Project Description </a></li>
    <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
    <li><a href="#setup-and-usage"> ➤ Setup and Usage</a></li>
    <li>
      <a href="#algorithms"> ➤ Algorithms</a>
      <ul>
        <li><a href="#linear-regression">Linear Regression</a></li>
      </ul>
    </li>
    <li><a href="#inspiration"> ➤ Inspiration</a></li>
  </ol>
</details>

## Project Description
Python implementation of a variety of Machine Learning (ML) algorithms from scratch, drawing <a href="#inspiration">inspiration</a> from various tutorials and training courses. This work involves the application of Object-Oriented Programming to produce supervised and unsupervised ML models, with the neccessary testing completed in order to demonstrate performance. Each ML algorithm is represented as an object that includes fit, predict and visualise methods to not only provide the functionality, but give the user a visual represetation of the statistical methods applied.

## Folder Structure

    ml-algorithms-from-scratch
    │
    ├── datasets
    │   ├── breast-cancer-wisconsin.data
    │   ├── breast-cancer-wisconsin.names
    │ 
    ├── ml_algorithms
    │   ├── __init__.py
    │   ├── k_means.py    
    │   ├── k_nearest_neighbours.py
    │   ├── linear_regression.py
    │   ├── mean_shift.py
    │   ├── support_vector_machine.py
    │ 
    ├── model_testing 
    │   ├── images
    │   │   ├── k_means_centroid_example.png
    │   │   ├── k_means_example.png
    │   │   ├── knn_boundary_example.png
    │   │   ├── knn_example.png
    │   │   ├── linear_regression_example.png
    │   │   ├── mean_shift_centroid_example.png
    │   │   ├── mean_shift_example.png
    │   │   ├── svm_boundary_example.png
    │   │   ├── svm_example.png
    │   ├── k_means_testing.py    
    │   ├── k_nearest_neighbours_testing.py
    │   ├── linear_regression_testing.py
    │   ├── mean_shift_testing.py
    │   ├── support_vector_machine_testing.py
    │ 
    ├── LICENSE 
    │ 
    ├── README.md 

## Setup and Usage

The following open source packages are used in this project:
* Numpy
* Matplotlib
* Scikit-Learn
* Random
* Warnings
* Collections
* Sys
* OS

The ML algorithms are individual modules contained within a pacakge named ml_algorithms. Given that the package has not yet been formally deployed, it is advised that the user takes the following steps:
<ol>
  <li>Clone the project and save to a local repository</li>
  <li>Add the ml_algorithms directory to the environment path</li>
  <li>Import the required machine learning algorithm</li>
</ol>

For example, to load the linear regression algorithm, the following code may be written within Python:

    import os
    import sys
    root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(root_folder)
    
    from ml_algorithms.linear_regression import LinearRegression

For convenience, the project directory contains an area to store data (<a href="https://github.com/jakeyk11/ml-algorithms-from-scratch/tree/main/datasets">datasets</a>) and an area to test ML algorithms (<a href="https://github.com/jakeyk11/ml-algorithms-from-scratch/tree/main/model_testing">model_testing</a>).

## Algorithms

### Linear Regression

## Inspiration

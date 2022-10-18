# An Open Mobile Communications Drive Test Data Set and its Use for Machine Learning #

This repository contains the code and data set used in the corresponding open access paper:
* <https://www.doi.org/10.1109/OJCOMS.2022.3210289>

Here is code that
* Provides an introduction to the data set (intro.ipynb run in an environment with req-geo.txt)
* Analyses the autocorrelation in time and space (spacetime.ipynb)
* Trains a neural network to predict the data rate (simply.py in an environment with req-ml.txt)
* Evaluates a hyperparameter search on the neural network (guild.yml and nn_results.ipynb in an environment with req-guild.txt)
* Uses LIME to explain the neural network (limeexplanation.py in an environment with req-lime.txt)
* Trains an autoencoder for signal strength parameters (autoencoder.py in an environment with req-ml.txt)

This repository is part of the Project 5G-AI-MLab of Salzburg Research: https://www.salzburgresearch.at/en/projekt/5g-ai-mlab/
The project is partially funded by the Austrian Federal Ministry of Climate Action, Environment, Energy, Mobility, Innovation and Technology (BMK) and the Austrian state Salzburg.

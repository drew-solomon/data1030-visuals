# DATA1030 Visualizations - 
ML Classification & Gradient Descent

This project involved the creation of interactive visualizations for machine learning
classification and gradient descent for the DATA1030 _Hands-on Data Science_  Brown Data Science Master’s curriculum. 

_(Practicum Project for DATA 2050 - Data Practicum @ Brown University Summer 2022)_

## Table of Contents
* [Project Description](#project-description)
* [Methods](#methods-used)
* [Technologies Used](#technologies-used)
* [Screenshots](#screenshots)
* [Demo](#demo)
* [Setup](#setup)
* [Usage](#usage)
* [Deliverables](#deliverables)
* [Contributing Members](#contributing-members)
* [Acknowledgements](#acknowledgements)


## Project Description

- **Why?** Visualizing ML classifications interactively can help students intuitively understand **how different ML classification models work** and **how different hyperparameters shape predictions**. Moreover, interactive classification plots allow students to easily view and experiment with **bias-variance trade-offs**.
- Similarly, visualizing gradient descent may help students intuitively understand **how gradient descent works** and the **role of the learning rate**.
- As a practicum project for DATA 2050, I created **interactive visualizations of classification ML models** and **interactive visualizations of gradient descent**, for use in the DATA 1030 Brown Data Science Master’s curriculum.
- For the interactive classification plots, I created functions to generate classification data (2D binary data for visualization) and to plot interactive visualizations _for any classification ML model_ with _any number of hyperparameters_ with any given ranges.
- For the gradient descent plots, I created functions to generate the gradient descent path data and to plot a two-part interactive plot of the gradient descent trajectory (over the cost function) along with a line chart of cost history.
- As **learning tools** for DATA1030, the classification visualizations illustrate different ML algorithms’ classifications as a
function of hyperparameters, their behavior with respect to outliers, nonlinearity, prediction
smoothness, and interpretability. In addition, the gradient descent plots show how the gradient
descent path and cost history vary as a function of learning rate, and when the algorithm fails to
converge.


## Methods Used
- Interactive data visualization/plotting
- Supervised ML algorithms
- Machine learning classification
	- Logistic Regression
    - Random Forest
    - Support Vector Machine (SVM)
    - XGBoost
    - K-Nearest Neighbors (KNN)
- Gradient descent (optimization)

## Technologies Used
- Python (3.10.4)
- Plotly (5.8.0)
- Ipywidgets (7.7.0)
- Pandas (1.4.2)
- Scikit-learn (1.1.1)
- Numpy (1.22.4)
- Jupyter Lab (3.4.2)
- Conda (4.14.0)

## Screenshots

### Classification Interactive Plots

The following screenshots illustrate example classification plots for a range of ML classification algorithms. The plots interactively show the given ML model’s classification predicted probabilities (as a contour plot) for toy 2D binary classification data. In particular, I chose the _make_moons_ dataset - with two interleaving half-circles over two features - for its clear visualization and nonlinearity. 

Each plot features sliders (for continuous hyperparameters) and dropdown menus (for categorical ones) for the given hyperparameter grid. Thus, the classification plots show how the classification probabilities and decision boundaries shift as the hyperparameters change, for any given ML classification model with any number of applicable hyperparameters and given ranges.

(_For each plot, the red points correspond to a class 0 and the blue points correspond to class 1. The contour colors correspond to the predicted probability of belonging to class 1. The black line shows the 0.5 probability decision boundary. As the hyperparameters are shifted, the plot contours, decision boundary, and title adjust accordingly._)



#### Support Vector Machine (SVM) 
|![SVC.png](https://i.postimg.cc/SxfSFcRK/SVC.png)|
|:--:|
|*Example interactive plot of SVM classification. Here, the sliders adjust hyperparameters gamma (kernel coefficient) and C (inverse of regularization strength).*|


#### Random Forest
|![RF.png](https://i.postimg.cc/52gVk1KY/RF.png)|
|:--:|
|*Example interactive plot of random forest classification. Here, the sliders adjust hyperparameters of split quality criterion (e.g. gini or log loss), the trees’ maximum depth, and the number of trees (i.e. estimators).*|

#### XGBoost 
|![XGBoost.png](https://i.postimg.cc/vZFFmTk6/XGBoost.png)|
|:--:|
|*Example interactive plot of XGBoost classification. Here, the sliders adjust hyperparameters of the trees’ maximum depth, alpha regularization (L1 regularization on leaf weights), and lambda regularization (L2 regularization on leaf weights).*|


#### K-nearest Neighbors (KNN)
|![KNN.png](https://i.postimg.cc/pTpRF7Z9/KNN.png)|
|:--:|
|*Example interactive plot of KNN classification. Here, the sliders adjust hyperparameters of the number of neighbors and how they’re weighted (uniformly or distance-based).*|

### Gradient Descent

The gradient descent plot is a two-part interactive plot. The first plot shows the cost history over the gradient descent iterations, and the second shows the gradient descent path over a contour plot of the cost function. For both plots, a single slider controls the learning rate. As the learning rate slider is changed, the gradient descent path and cost history are updated.  
 
#### Gradient Descent Cost History and Path - Interactive Plot
|![Gradient-Descent.png](https://i.postimg.cc/4NTRfnrH/Gradient-Descent.png)|
|:--:|
|*Example interactive plot of gradient descent.*|

## Demo

### Classification Plot - Random Forest Example


https://user-images.githubusercontent.com/71301990/206095016-43e35ad9-d2b6-47ea-b412-eec1b119c05b.mov

### Gradient Descent Plot


https://user-images.githubusercontent.com/71301990/206095579-23edfd39-b081-4655-a5fc-3c1c33be48a9.mov




## Setup

#### Read the project notebooks
To read the classification plot notebook as a Jupyter/IPython notebook, simply click on the [classification_plots.ipynb](https://github.com/BrownDSI/data1030-visuals/blob/main/classification_plots.ipynb) file to open it in your browser.

Similarly, to read the gradient descent notebook as a Jupyter/IPython notebook, simply click on the [gradient_descent.ipynb](https://github.com/BrownDSI/data1030-visuals/blob/main/gradient_descent.ipynb) file to open it in your browser.


#### Run the interactive plots

To run the interactive plots, first set up the project conda environment and open the plotting notebook with Jupyter with the following step-by-step instructions:

1. Clone this project github repo.
2. Copy and activate the project conda environment (with [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)) by running the following commands in a terminal:
    ```console
    conda env create -f data1030vis.yml
    conda activate data1030vis
    ```
3. If Jupyter is not already installed, run:
    ```console
    conda install jupyter
    ```
    With Jupyter installed, start a Jupyter session in your browser by running:
    ```console
    jupyter notebook
    ```
4. Then, in the Jupyter browser, open the desired plot notebook:
    - For the interactive classification plots, open the  _"classification_plots.ipynb"_ notebook. 
    - For the interactive gradient descent plots, open the  _"gradient_descent.ipynb"_ notebook. 

    To run all the code cells, go to _Cell -> Run All_ in the top menu. Done!


## Usage

### Classification Plots

To use the classification plots, move the sliders (and menu options for categorical hyperparameters) to see how changes in the hyperparameters change the ML model’s predicted probabilities and decision boundary! 
> How do different models compare? Are predictions nonlinear? Smooth? For what hyperparameter values does the model underfit or overfit?

To customize your own interactive classification plot, generate the plot data (as a dictionary of 2D arrays of predicted probabilities) using the ```generate_all_data(X, y, ml_algo, param_grid, h)``` function, choosing your own:
- Input classification data (2D binary data - e.g. new patterns or adjust noise)
- ML classification algorithm
- Hyperparameter grid (any number of hyperparameters with any valid ranges, continuous or categorical)
- _(Contour plot grid step size - h)_

Then, plot the interactive plot (which uses Plotly FigureWidgets) using the ```plot_clf_contour(param_grid, X, y, Z_dict, h)``` function (where ```Z_dict``` is the generated plot data).

### Gradient Descent Plots

To use the gradient descent two-part plot, move the slider to see how changes in learning rate impact the gradient descent path (over the cost function) as well as the cost history.
> How quickly does the algorithm converge? For what learning rates does the algorithm converge the fastest (for the given cost function), and when does it fail to converge?

To customize your own interactive gradient plot, generate the plot data (as a dictionary of theta and cost histories) using the ```generate_grad_desc_data(X,y,theta,learning_rates,iterations=100)``` function, choosing your own:
- Input regression data (1 feature for plot, e.g. new regression problem or adjust noise)
- Vector of learning rates
- Initial theta (slope) range

Then, plot the interactive plot by running the _Double Plot - Cost History and Gradient Descent Path_ code cell.

## Deliverables
- Project [pitch](https://drive.google.com/file/d/1BHirPB2A9imPciSByesbvf_YLWRCzvjH/view?usp=share_link)
- Practicum [presentation](https://drive.google.com/file/d/11NFRN_huLnMthzg2k7VwmmR7FVUbDTBQ/view?usp=share_link) and [slides](https://drive.google.com/file/d/11NFRN_huLnMthzg2k7VwmmR7FVUbDTBQ/view?usp=share_link)
- Final project [report](https://github.com/drew-solomon/data1030-visuals/blob/main/report/practicum_report.pdf)

## Contributing Members
- [Drew Solomon](https://github.com/drew-solomon) (me)
- ([Yifei Song](https://github.com/ysong1020) contributed on a related project - regression visualization - in the original Brown DSI [repo](https://github.com/BrownDSI/data1030-visuals).)


## Acknowledgements
- This project builds off and extends Brown University’s Data Science Master’s _“Hands-on Data Science”_ DATA1030 curriculum designed by Professor Andras Zsom, particularly on the supervised ML algorithms and gradient descent sections. 
- Many thanks to Professor Andras Zsom for this opportunity and your supervision - I hope the next DATA1030 students enjoy!

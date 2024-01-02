# Survival Analysis

- Author: [Lina Faik](https://www.linkedin.com/in/lina-faik/)
- Creation date: June 2023
- Last update: June 2023

## Objective

This repository contains the code and the notebook used to train a DLRM model using PyTorch library, [torchrec](https://pytorch.org/torchrec/).
It was developed as an experimentation project to support the explanation blog posts around the topic. 

This repository contains the code and notebook used to train a DLRM model using the PyTorch library, [torchrec](https://pytorch.org/torchrec/). 
The model and code are explained in more detail in the following article:
- [Building Powerful Recommender Systems with Deep Learning  
_A Step-by-Step Implementation Using the PyTorch Library TorchRec]([link coming soon](https://towardsdatascience.com/building-powerful-recommender-systems-with-deep-learning-d8a919c52119))_

<div class="alert alert-block alert-info"> You can find all my technical blog posts <a href = https://linafaik.medium.com/>here</a>. </div>

## Project Description

### Code structure

```
notebook.ipynb # central code where a DLRM model is trained and evaluated using synthetical data
src
├── batch.py # general functions used to build batch from raw data        
├── model.py # model related class and functions
```

### Data

The notebook is based on synthetic data. 
It can easily be adapted to any other data set with categorical and continuous variables and a binary target to predict.

## How to Use This Repository?

### Requirement

The code uses a GPU and relies on the following libraries:

```
torch
torchrec==0.4.0
plotly==5.13.1
sklearn-pandas==2.2.0
```

### Experiments

To run experiments, you need to run the notebook `notebook.ipynb`.
The associated code is in the `src` directory.

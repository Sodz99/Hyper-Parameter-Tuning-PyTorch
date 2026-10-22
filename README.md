# Hyperparameter Tuning for Neural Networks with PyTorch

- To understand the hyperparameters of neural networks - To understand how to tune hyperparameters for improving model performance in PyTorch

## Overview
Hyperparameters are a set of parameters whose value controls the learning process of the model. The performance of models can be greatly improved by tuning their hyperparameters. Tuning hyperparameters means you are trying to find out the set of optimal parameters, giving you better performance than the default hyperparameters of the model. In the previous project of the series Learn How to Build Neural Networks from Scratch, we saw what Neural Networks are and how we can build a Neural Network for the classification model in Pytorch. In this project, we will tune the hyperparameters such as learning rate, epochs, dropout, Early stopping, checkpoints to improve the performance of our classification model.

## Data
The dataset used in this project has information about the customer churn based on the various features. The dataset contains 2000 rows and 15 features to predict the churn.

_Data is not included in this repository._

## Approach
- Data cleaning
- Data preprocessing
- Building a sequential neural network
- Model training
- Tuning hyperparameters

Modular code overview:

Once you unzip the Neural_network_tuning.zip file, you can find the following folders within it.
1. Input
2. ML_Pipeline
3. model
4. Notebook
5. Output
6. Engine.py
7. Readme.md
8. requirements.txt
1. The Input folder contains the data that we have for analysis. In our case, it contains data.csv
2. The Notebook folder contains the jupyter notebook file of the project.
3. The model folder contains the checkpoints of the jupyter notebook file
4. The Output folder contains the saved checkpoints of the model.
5. The ML_pipeline is a folder that contains all the functions put into different python files, which are appropriately named. These python functions are then called inside the Engine.py file.
6. The requirements.txt file has all the required libraries with respective versions. Kindly install the file by using the command pip install -r requirements.txt
7. All the instructions for running the code are present in Readme.md file

## Tech stack
➔ Language: Python ➔ Libraries: Pandas, Pytorch,matplotlib,sci-kit learn, NumPy, imblearn

## How to run
```bash
pip install -r Neural_network_tuning/requirements.txt
```
Notebooks: [`NeuralNetHyperparameters.ipynb`](NeuralNetHyperparameters.ipynb)

## Project structure
```
.gitignore
NeuralNetHyperparameters.ipynb
Neural_network_tuning.zip
Neural_network_tuning/
  Engine.py
  MLPipeline/
    Dropout.py
    EarlyStopping.py
    NeuralNet.py
    Preprocessing.py
    Regularization.py
    SaveAndLoad.py
  Readme.md
  requirements.txt
README.md
Solution Methodology.pdf
data.zip
```

## Credits
Based on a ProjectPro guided project.

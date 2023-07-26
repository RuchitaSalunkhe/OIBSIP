# Iris Flower Classification with SVM

This project aims to classify iris flowers into three species: setosa, versicolor, and virginica based on their measurements (sepal length, sepal width, petal length, and petal width). The model is trained using Support Vector Machine (SVM), a popular machine learning algorithm for classification tasks.

# Dataset
The dataset used for this project is the famous Iris dataset, which is Provided by organization. The dataset contains 150 samples of iris flowers, each with four features and one target label (species).

# Features
The feature dataset (x) consists of four measurements for each sample:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)
- Target

The target dataset (y) contains the encoded species labels for each sample:

- 0: setosa
- 1: versicolor
- 2: virginica

# Model
The SVM classifier is used with a linear kernel (kernel='linear') and regularization parameter C set to 1.0 (C=1.0)

# Evaluation
The model's performance is evaluated on a test dataset using the following metrics:

- Accuracy: The proportion of correctly classified samples.

- Classification Report: Includes precision, recall, and F1-score for each class.

- Confusion Matrix: A matrix showing the count of true positives, false positives, true negatives, and false negatives for each class.

# Deployment
To deploy this project run use following important python libraries

1. pandas (import pandas as pd):
pandas is a powerful library for data manipulation and analysis in Python. It is used to handle the iris dataset, load the data from CSV files, and manage the data in DataFrames.

2. scikit-learn (from sklearn.model_selection import train_test_split, from sklearn.svm import SVC, from sklearn.metrics import accuracy_score, classification_report, confusion_matrix):
- scikit-learn is a widely used library for machine learning in Python.
- It provides tools for data preprocessing, model selection, and evaluation.
- train_test_split: This function is used to split the dataset into training and testing sets for model evaluation.
- SVC: It stands for Support Vector Classifier, and it's the implementation of the SVM algorithm for classification tasks.
- accuracy_score: This function is used to calculate the accuracy of the model's predictions.
- classification_report: It provides a comprehensive report of precision, recall, and F1-score for each class.
- confusion_matrix: This function is used to generate the confusion matrix to evaluate the model's performance.
3. NumPy (not explicitly imported):
NumPy is a fundamental package for numerical computing in Python.
pandas internally uses NumPy arrays to store data efficiently.
4. Optional Libraries:
There are two optional libraries that may be used for additional functionality:
- matplotlib: It can be used for data visualization. However, it is not explicitly used in the provided code.
- seaborn: It can be used for statistical data visualization. Similar to matplotlib, it is not used in the provided code.
```bash
pip install pandas scikit-learn
```

# Installation
1. Install my-project by using any python compailer

2. Then install following python libraries

```bash
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```
3. Import Iris Dataset 
4. After importing dataset execute code which provide in repository

# Result

The SVM model achieves high accuracy in classifying iris flowers into their respective species. The classification report and confusion matrix provide detailed performance metrics for each class.

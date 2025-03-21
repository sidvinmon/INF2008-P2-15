# User Manual for INF2008 Machine Learning Project

## Overview

This manual provides step-by-step instructions to reproduce the results of the Sentiment Analysis Project using a Jupyter Notebook running in Google Colaboratory. The tasks include setting up the environment, processing data, and training/testing machine learning models.

## 1. Environment Setup

### Requirements

This project is designed to run on Google Colab. The following libraries must be installed:

```sh
pip install contractions
pip install lxml
pip install html-clean
pip install py-readability-metrics
pip install gradio
```

### Libraries Used

- pandas, numpy, matplotlib, seaborn
- nltk, textblob, readability
- scikit-learn, xgboost, lightgbm, hyperopt, tqdm
- pickle

### Download necessary NLTK resources

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

### Google Drive Setup (if using Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

Place your dataset under a specific path (e.g., `/content/drive/MyDrive/YourFolder/your_dataset.csv`).

## 2. Data Processing

### Dataset Overview

The dataset should contain user reviews and corresponding sentiment labels (positive, negative, etc.).

### Preprocessing Steps

- Expand contractions (isn't → is not)
- Remove HTML tags, links, and special characters
- Tokenization: split into words and sentences
- Lemmatization and POS tagging using NLTK

### Feature Engineering

- Word Count
- Sentence Count
- Capital Word Ratio
- Count of Exclamation Marks, Question Marks, and Ellipses
- Readability Score using py-readability-metrics
- POS-based features: Adjective Count, Adverb Count, Noun/Verb Ratio

All engineered features are stored in a Pandas DataFrame and standardized before modeling.

## 3. Model Training & Testing

### Train/Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
```

### Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- AdaBoost
- XGBoost
- LightGBM

### Hyperparameter Tuning

Used hyperopt and RandomizedSearchCV for automatic tuning:

```python
from hyperopt import fmin, tpe, hp, Trials
```

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

Visualizations were generated using matplotlib and seaborn.

## 4. Saving & Reusing Models

### To save a model

```python
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### To load a model

```python
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

## 5. Reproducibility Tips

- Use `random_state=42` to ensure consistent splits
- Set seeds for NumPy and random
- Install and download all NLTK resources before execution
- Always standardize/scale data before model fitting
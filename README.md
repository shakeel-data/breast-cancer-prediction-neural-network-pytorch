# 🧠 Breast-cancer-prediction-nn-pytorch
![image](https://github.com/user-attachments/assets/091e92d8-9414-4edc-bd2c-5981ee215fa5)

Breast cancer is a leading cause of death among women, where early and accurate detection is vital for improving survival rates. Traditional methods are often costly, time-consuming, and error-prone. Using machine learning and neural networks, we can build models that offer faster, more reliable, and scalable diagnosis. This enhances clinical decision-making, supports early intervention, and makes AI a powerful tool in life-saving healthcare applications.

## 📘 Project Overview
This project focuses on breast cancer prediction using a neural network implemented with PyTorch. It uses a structured dataset containing quantitative features extracted from digitized fine needle aspirate (FNA) test results, such as cell radius, texture, and perimeter. The objective is to classify tumors as malignant (cancerous) or benign (non-cancerous). The process includes data preprocessing, feature scaling, model design, training, and evaluation. This project showcases how deep learning can support early diagnosis and improve clinical outcomes through data-driven decision-making in healthcare.

## 🎯 Key Objectives
- **Understand Data:** Explore the **Breast Cancer Wisconsin dataset** to identify key features and their distributions.
- **Preprocessing:** Apply feature standardization to ensure balanced input for **neural network training.**
- **Model Development:** Build a **PyTorch-based neural network** to learn complex patterns for classification.
- **Training:** Train the model using **Binary Cross-Entropy loss and Adam** optimizer for optimal learning.
- **Evaluation:** Assess performance using metrics like accuracy on both **training and test datasets.**
- **Achieve High Accuracy:** Demonstrate strong predictive reliability for **real-world diagnostic support.**

## 📁 Data Sources
- Kaggle
  <a href="">csv</a>
- Python
  <a href="">codes</a>

## 🔧 Project Workflow
### 1. Importing Dependencies and Data load

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```
```python
df = pd.read_csv("file-path")
print('Read sucessfully')
```

### 2. Device configuration

```python
# check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### 3. Data collection and Preprocessing

```python
print("Breast cancer data -  rows:",df.shape[0]," columns:", df.shape[1])
```
![image](https://github.com/user-attachments/assets/5a6f2ec8-026c-46c9-8040-2a99c663dbce)

**Data type**
```python
df.info()
```
**<class 'pandas.core.frame.DataFrame'>
RangeIndex: 569 entries, 0 to 568
Data columns (total 33 columns):**

| **Column**                | **Non-Null Count** | **Dtype** |
| ------------------------- | ------------------ | --------- |
| id                        | 569                | int64     |
| diagnosis                 | 569                | object    |
| radius\_mean              | 569                | float64   |
| texture\_mean             | 569                | float64   |
| perimeter\_mean           | 569                | float64   |
| area\_mean                | 569                | float64   |
| smoothness\_mean          | 569                | float64   |
| compactness\_mean         | 569                | float64   |
| concavity\_mean           | 569                | float64   |
| concave points\_mean      | 569                | float64   |
| symmetry\_mean            | 569                | float64   |
| fractal\_dimension\_mean  | 569                | float64   |
| radius\_se                | 569                | float64   |
| texture\_se               | 569                | float64   |
| perimeter\_se             | 569                | float64   |
| area\_se                  | 569                | float64   |
| smoothness\_se            | 569                | float64   |
| compactness\_se           | 569                | float64   |
| concavity\_se             | 569                | float64   |
| concave points\_se        | 569                | float64   |
| symmetry\_se              | 569                | float64   |
| fractal\_dimension\_se    | 569                | float64   |
| radius\_worst             | 569                | float64   |
| texture\_worst            | 569                | float64   |
| perimeter\_worst          | 569                | float64   |
| area\_worst               | 569                | float64   |
| smoothness\_worst         | 569                | float64   |
| compactness\_worst        | 569                | float64   |
| concavity\_worst          | 569                | float64   |
| concave points\_worst     | 569                | float64   |
| symmetry\_worst           | 569                | float64   |
| fractal\_dimension\_worst | 569                | float64   |
| Unnamed: 32               | 0                  | float64   |
**dtypes: float64(31), int64(1), object(1)**

**Drop the column**
```python
df.drop("Unnamed: 32", axis=1, inplace=True)
```

**First few rows of Data**
```python
df.head()
```
| id       | diagnosis | radius\_mean | texture\_mean | perimeter\_mean | area\_mean | smoothness\_mean | compactness\_mean | concavity\_mean | concave points\_mean | ... | texture\_worst | perimeter\_worst | area\_worst | smoothness\_worst | compactness\_worst | concavity\_worst | concave points\_worst | symmetry\_worst | fractal\_dimension\_worst |
| -------- | --------- | ------------ | ------------- | --------------- | ---------- | ---------------- | ----------------- | --------------- | -------------------- | --- | -------------- | ---------------- | ----------- | ----------------- | ------------------ | ---------------- | --------------------- | --------------- | ------------------------- |
| 842302   | M         | 17.99        | 10.38         | 122.80          | 1001.0     | 0.11840          | 0.27760           | 0.3001          | 0.14710              | ... | 17.33          | 184.60           | 2019.0      | 0.1622            | 0.6656             | 0.7119           | 0.2654                | 0.4601          | 0.11890                   |
| 842517   | M         | 20.57        | 17.77         | 132.90          | 1326.0     | 0.08474          | 0.07864           | 0.0869          | 0.07017              | ... | 23.41          | 158.80           | 1956.0      | 0.1238            | 0.1866             | 0.2416           | 0.1860                | 0.2750          | 0.08902                   |
| 84300903 | M         | 19.69        | 21.25         | 130.00          | 1203.0     | 0.10960          | 0.15990           | 0.1974          | 0.12790              | ... | 25.53          | 152.50           | 1709.0      | 0.1444            | 0.4245             | 0.4504           | 0.2430                | 0.3613          | 0.08758                   |
| 84348301 | M         | 11.42        | 20.38         | 77.58           | 386.1      | 0.14250          | 0.28390           | 0.2414          | 0.10520              | ... | 26.50          | 98.87            | 567.7       | 0.2098            | 0.8663             | 0.6869           | 0.2575                | 0.6638          | 0.17300                   |
| 84358402 | M         | 20.29        | 14.34         | 135.10          | 1297.0     | 0.10030          | 0.13280           | 0.1980          | 0.10430              | ... | 16.67          | 152.20           | 1575.0      | 0.1374            | 0.2050             | 0.4000           | 0.1625                | 0.2364          | 0.07678                   |





























































































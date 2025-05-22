# 🎗 Breast Cancer Prediction Project | Neural Network + PyTorch
![image](https://github.com/user-attachments/assets/091e92d8-9414-4edc-bd2c-5981ee215fa5)

Breast cancer is a leading cause of **death among women**, where early and accurate detection is vital for improving survival rates. Traditional methods are often costly, time-consuming, and error-prone. Using machine learning and neural networks, we can build models that **offer faster, more reliable, and scalable diagnosis.** This enhances clinical decision-making, supports early intervention, and makes AI a powerful tool in life-saving healthcare applications.

## 📘 Project Overview
This project focuses on breast cancer prediction using a neural network built in PyTorch within the **Spyder IDE environment**. It uses a structured dataset containing quantitative features extracted from digitized fine needle aspirate (FNA) test results, such as cell radius, texture, and perimeter.The workflow includes **data preprocessing, feature scaling, model design, training, and evaluation.** This project demonstrates the power of **deep learning** in enabling early and accurate diagnosis, contributing to improved **clinical outcomes in healthcare**.

## 🎯 Key Objectives
- **Understand Data:** Explore the **Breast Cancer Wisconsin dataset** to identify key features and their distributions.
- **Preprocessing:** Apply feature standardization to ensure balanced input for **neural network training.**
- **Model Development:** Build a **PyTorch-based neural network** to learn complex patterns for classification.
- **Training:** Train the model using **Binary Cross-Entropy loss and Adam** optimizer for optimal learning.
- **Evaluation:** Assess performance using metrics like accuracy on both **training and test datasets.**
- **Achieve High Accuracy:** Demonstrate strong predictive reliability for **real-world diagnostic support.**

## 📁 Data Sources
- Kaggle
  <a href="https://github.com/shakeel-data/Breast-cancer-prediction-nn-pytorch/blob/main/Clinical_data.csv">csv</a>
- Python
  <a href="https://github.com/shakeel-data/Breast-cancer-prediction-nn-pytorch/blob/main/Breast%20cancer%20prediction%20using%20neural%20network%20in%20pytorch.py">codes</a>

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

### 2. 💻 Device configuration

```python
# check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### 3. 🗂️ Data collection and Preprocessing

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

**5 rows × 32 columns**

**Last few rows of Data**
```python
df.tail()
```

| id     | diagnosis | radius\_mean | texture\_mean | perimeter\_mean | area\_mean | smoothness\_mean | compactness\_mean | concavity\_mean | concave points\_mean | ... | texture\_worst | perimeter\_worst | area\_worst | smoothness\_worst | compactness\_worst | concavity\_worst | concave points\_worst | symmetry\_worst | fractal\_dimension\_worst |
| ------ | --------- | ------------ | ------------- | --------------- | ---------- | ---------------- | ----------------- | --------------- | -------------------- | --- | -------------- | ---------------- | ----------- | ----------------- | ------------------ | ---------------- | --------------------- | --------------- | ------------------------- |
| 926424 | M         | 21.56        | 22.39         | 142.00          | 1479.0     | 0.11100          | 0.11590           | 0.24390         | 0.13890              | ... | 26.40          | 166.10           | 2027.0      | 0.14100           | 0.21130            | 0.4107           | 0.2216                | 0.2060          | 0.07115                   |
| 926682 | M         | 20.13        | 28.25         | 131.20          | 1261.0     | 0.09780          | 0.10340           | 0.14400         | 0.09791              | ... | 38.25          | 155.00           | 1731.0      | 0.11660           | 0.19220            | 0.3215           | 0.1628                | 0.2572          | 0.06637                   |
| 926954 | M         | 16.60        | 28.08         | 108.30          | 858.1      | 0.08455          | 0.10230           | 0.09251         | 0.05302              | ... | 34.12          | 126.70           | 1124.0      | 0.11390           | 0.30940            | 0.3403           | 0.1418                | 0.2218          | 0.07820                   |
| 927241 | M         | 20.60        | 29.33         | 140.10          | 1265.0     | 0.11780          | 0.27700           | 0.35140         | 0.15200              | ... | 39.42          | 184.60           | 1821.0      | 0.16500           | 0.86810            | 0.9387           | 0.2650                | 0.4087          | 0.12400                   |
| 92751  | B         | 7.76         | 24.54         | 47.92           | 181.0      | 0.05263          | 0.04362           | 0.00000         | 0.00000              | ... | 30.37          | 59.16            | 268.6       | 0.08996           | 0.06444            | 0.0000           | 0.0000                | 0.2871          | 0.07039                   |

**5 rows × 32 columns**

**Checking Null values**
```python
df.isnull().sum()
```

| Column                    | Null Count |
| ------------------------- | ---------- |
| id                        | 0          |
| diagnosis                 | 0          |
| radius\_mean              | 0          |
| texture\_mean             | 0          |
| perimeter\_mean           | 0          |
| area\_mean                | 0          |
| smoothness\_mean          | 0          |
| compactness\_mean         | 0          |
| concavity\_mean           | 0          |
| concave points\_mean      | 0          |
| symmetry\_mean            | 0          |
| fractal\_dimension\_mean  | 0          |
| radius\_se                | 0          |
| texture\_se               | 0          |
| perimeter\_se             | 0          |
| area\_se                  | 0          |
| smoothness\_se            | 0          |
| compactness\_se           | 0          |
| concavity\_se             | 0          |
| concave points\_se        | 0          |
| symmetry\_se              | 0          |
| fractal\_dimension\_se    | 0          |
| radius\_worst             | 0          |
| texture\_worst            | 0          |
| perimeter\_worst          | 0          |
| area\_worst               | 0          |
| smoothness\_worst         | 0          |
| compactness\_worst        | 0          |
| concavity\_worst          | 0          |
| concave points\_worst     | 0          |
| symmetry\_worst           | 0          |
| fractal\_dimension\_worst | 0          |

**dtype: int64**

**Statistical information**
```python
df.describe()
```

| Statistic | id       | radius\_mean | texture\_mean | perimeter\_mean | area\_mean  | smoothness\_mean | compactness\_mean | concavity\_mean | concave points\_mean | symmetry\_mean | ... | radius\_worst | texture\_worst | perimeter\_worst | area\_worst | smoothness\_worst | compactness\_worst | concavity\_worst | concave points\_worst | symmetry\_worst | fractal\_dimension\_worst |
| --------- | -------- | ------------ | ------------- | --------------- | ----------- | ---------------- | ----------------- | --------------- | -------------------- | -------------- | --- | ------------- | -------------- | ---------------- | ----------- | ----------------- | ------------------ | ---------------- | --------------------- | --------------- | ------------------------- |
| **count** | 5.69e+02 | 569.000000   | 569.000000    | 569.000000      | 569.000000  | 569.000000       | 569.000000        | 569.000000      | 569.000000           | 569.000000     | ... | 569.000000    | 569.000000     | 569.000000       | 569.000000  | 569.000000        | 569.000000         | 569.000000       | 569.000000            | 569.000000      | 569.000000                |
| **mean**  | 3.04e+07 | 14.127292    | 19.289649     | 91.969033       | 654.889104  | 0.096360         | 0.104341          | 0.088799        | 0.048919             | 0.181162       | ... | 16.269190     | 25.677223      | 107.261213       | 880.583128  | 0.132369          | 0.254265           | 0.272188         | 0.114606              | 0.290076        | 0.083946                  |
| **std**   | 1.25e+08 | 3.524049     | 4.301036      | 24.298981       | 351.914129  | 0.014064         | 0.052813          | 0.079720        | 0.038803             | 0.027414       | ... | 4.833242      | 6.146258       | 33.602542        | 569.356993  | 0.022832          | 0.157336           | 0.208624         | 0.065732              | 0.061867        | 0.018061                  |
| **min**   | 8.67e+03 | 6.981000     | 9.710000      | 43.790000       | 143.500000  | 0.052630         | 0.019380          | 0.000000        | 0.000000             | 0.106000       | ... | 7.930000      | 12.020000      | 50.410000        | 185.200000  | 0.071170          | 0.027290           | 0.000000         | 0.000000              | 0.156500        | 0.055040                  |
| **25%**   | 8.69e+05 | 11.700000    | 16.170000     | 75.170000       | 420.300000  | 0.086370         | 0.064920          | 0.029560        | 0.020310             | 0.161900       | ... | 13.010000     | 21.080000      | 84.110000        | 515.300000  | 0.116600          | 0.147200           | 0.114500         | 0.064930              | 0.250400        | 0.071460                  |
| **50%**   | 9.06e+05 | 13.370000    | 18.840000     | 86.240000       | 551.100000  | 0.095870         | 0.092630          | 0.061540        | 0.033500             | 0.179200       | ... | 14.970000     | 25.410000      | 97.660000        | 686.500000  | 0.131300          | 0.211900           | 0.226700         | 0.099930              | 0.282200        | 0.080040                  |
| **75%**   | 8.81e+06 | 15.780000    | 21.800000     | 104.100000      | 782.700000  | 0.105300         | 0.130400          | 0.130700        | 0.074000             | 0.195700       | ... | 18.790000     | 29.720000      | 125.400000       | 1084.000000 | 0.146000          | 0.339100           | 0.382900         | 0.161400              | 0.317900        | 0.092080                  |
| **max**   | 9.11e+08 | 28.110000    | 39.280000     | 188.500000      | 2501.000000 | 0.163400         | 0.345400          | 0.426800        | 0.201200             | 0.304000       | ... | 36.040000     | 49.540000      | 251.200000       | 4254.000000 | 0.222600          | 1.058000           | 1.252000         | 0.291000              | 0.663800        | 0.207500                  |

**8 rows × 31 columns**

**Load the breast cancer dataset**
```python
data = load_breast_cancer()
X = data.data
y = data.target
```
```python
print(X)
```
![image](https://github.com/user-attachments/assets/170e3cc3-1627-4969-902f-b5a420dced5b)

```python
print(y)
```
![image](https://github.com/user-attachments/assets/fc55d25b-5df4-4fab-9142-42d47ec794f4)


### 4. 📊 Exploratory Data Analysis (EDA)

**Heatmap**
```python
plt.figure(figsize=(20, 10))
sns.heatmap(df.describe().T, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Statistical Summary of Numerical Columns")
plt.show()
```
![image](https://github.com/user-attachments/assets/35dfb34a-53d3-4eff-b114-5606aac4b1f7)

**Pairplot**
```python
sns.pairplot(df[['radius_mean', 'texture_mean', 'perimeter_mean','area_mean', 'smoothness_mean']])
plt.show()
```
![image](https://github.com/user-attachments/assets/10abf6ea-3852-4310-8857-bb5b3c6a86bd)

**Histogram**
```python
scaler = StandardScaler()
# Fit and transform the 'radius_mean' column 
scaled_data = scaler.fit_transform(df[['radius_mean']]) 

# Create a new DataFrame with the scaled data for 'radius_mean'
scaled_df = pd.DataFrame(scaled_data, columns=['radius_mean'], index=df.index)

plt.figure()
plt.hist(df['radius_mean'], alpha=0.5, label='Raw')
plt.hist(scaled_df['radius_mean'], alpha=0.5, label='Scaled') # Now scaled_df is defined
plt.legend()
plt.title('Feature Scaling: radius_mean')
plt.show()
```
![image](https://github.com/user-attachments/assets/a1681487-1eb4-4105-a337-08915dc19900)

**Confusion matrix**
```python
y_true = np.array([0, 1, 0, 1, 1])  # Example ground truth
y_pred = np.array([1, 1, 0, 0, 1])  # Example predictions

cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)  # Assuming binary classification
plt.xticks(tick_marks, ['Class 0', 'Class 1'], rotation=45)
plt.yticks(tick_marks, ['Class 0', 'Class 1'])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
```
![image](https://github.com/user-attachments/assets/46ba2a39-1751-4ef6-bf0a-446a962e3f48)

### 5. ✂️  Split the dataset into training and test set
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
```python
print(X.shape)
print(X_train.shape)
print(X_test.shape)
```
![image](https://github.com/user-attachments/assets/bb1c11be-5325-40f7-9aa0-504629b35b65)

### 6. 📐 Standardize the data using Standard sclaer
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
```python
type(X_train)
```

**Convert data to PyTorch tensors and move it to GPU**
```python
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
```

### 7. 🧩 Neural Network Architecture
```python
class NeuralNet(nn.Module):

  def __init__(self, input_size, hidden_size, output_size):
    super(NeuralNet, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, output_size)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    out = self.sigmoid(out)
    return out
```

### 8. ⚙️ Hyperparameters
```python
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 100
```

**nitialize the Neural Network and move it the GPU**
```python
model = NeuralNet(input_size, hidden_size, output_size).to(device)
```

**Loss and the Optiizer**
```python
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

### 9. 🔄 Training the Neural Network
```python
# training the model
for epoch in range(num_epochs):
  model.train()
  optimizer.zero_grad()
  outputs = model(X_train)
  loss = criterion(outputs, y_train.view(-1,1))
  loss.backward()
  optimizer.step()

  # claculate accuracy
  with torch.no_grad():
    predicted = outputs.round()
    correct = (predicted == y_train.view(-1,1)).float().sum()
    accuracy = correct/y_train.size(0)

  if (epoch+1) % 10 == 0:
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss : {loss.item():.4f}, Accuracy: {accuracy.item() * 100:.2f}%")
```
![image](https://github.com/user-attachments/assets/f6c8a1c3-bf2c-488c-881e-d5aa55700db0)


### 10. 🤖 Model Evaluation
**Evaluation on training set**
```python
model.eval()
with torch.no_grad():
  outputs = model(X_train)
  predicted = outputs.round()
  correct = (predicted == y_train.view(-1,1)).float().sum()
  accuracy = correct/y_train.size(0)
  print(f"Accuracy on training data: {accuracy.item() * 100:.2f}%")
```
![image](https://github.com/user-attachments/assets/60adf77a-f4c6-40ba-a489-f84e36955380)

**Evaluation on test set**
```python
model.eval()
with torch.no_grad():
  outputs = model(X_test)
  predicted = outputs.round()
  correct = (predicted == y_test.view(-1,1)).float().sum()
  accuracy = correct/y_test.size(0)
  print(f"Accuracy on test data: {accuracy.item() * 100:.2f}%")
```
![image](https://github.com/user-attachments/assets/3377ccbe-d4b4-435d-876f-8ec2cdce3640)

## 🌟 Key Insights
- **High Predictive Accuracy:** The neural network achieved **98.02%** accuracy on the training set and **97.37%** on the test set, indicating excellent generalization to new data.
- **Model Effectiveness:** A simple architecture with one hidden layer was sufficient for achieving **high classification accuracy.**
- **Preprocessing Importance:** Standardizing features with **StandardScaler improved model stability and performance** by preventing large-range attributes from dominating the learning process.
- **PyTorch Efficiency:** Using PyTorch, along with **GPU acceleration (CUDA),** accelerated training over 100 epochs, making the process more efficient.

## ☁️ Technologies and Tools
- **Kaggle** – Dataset source
- **Spyder IDE** – Interactive environment for coding and presenting analysis
- **Python**
  - Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`
- **Machine Learning** – Model development and evaluation
  - Scikit-learn: `train_test_split`, `StandardScaler`
- **Deep Learning** – Neural network 
  - PyTorch: `torch`, `torch.nn`, `torch.optim`

## ✅ Conclusion
This project successfully developed a **PyTorch-based neural network for classifying breast cancer tumors**. Through a structured workflow of data preprocessing, model training, and evaluation, the model achieved over 97% accuracy on the test dataset. This result highlights the **power of neural networks, even simple architectures,** in addressing complex medical classification tasks when applied to relevant features. 

The project demonstrates how deep learning tools can aid in **medical diagnosis, showcasing the potential of reliable predictive models based on quantitative image-derived data**. The model’s strong generalization suggests it could be valuable in **computer-aided breast cancer screening systems.**

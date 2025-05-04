                                        # Importing Dependencies
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


                                            # Read the data
df = pd.read_csv("file-path")
print('Read sucessfully')


                                        # Device configuration
## check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


                                 # Data collection and Preprocessing
print("Breast cancer data -  rows:",df.shape[0]," columns:", df.shape[1])

### Data type
df.info()

### Drop the column
df.drop("Unnamed: 32", axis=1, inplace=True)

### First few rows of Data
df.head()

### Last few rows of Data
df.tail()

### Checking Null values
df.isnull().sum()

### Statistical information
df.describe()

### load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

print(X)
print(y)


                                    # Exploratory Data Analysis (EDA)
## Heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(df.describe().T, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Statistical Summary of Numerical Columns")
plt.show()


## Pairplot
sns.pairplot(df[['radius_mean', 'texture_mean', 'perimeter_mean','area_mean', 'smoothness_mean']])
plt.show()


## Histogram
scaler = StandardScaler()
### Fit and transform the 'radius_mean' column 
scaled_data = scaler.fit_transform(df[['radius_mean']]) 

### Create a new DataFrame with the scaled data for 'radius_mean'
scaled_df = pd.DataFrame(scaled_data, columns=['radius_mean'], index=df.index)

plt.figure()
plt.hist(df['radius_mean'], alpha=0.5, label='Raw')
plt.hist(scaled_df['radius_mean'], alpha=0.5, label='Scaled') # Now scaled_df is defined
plt.legend()
plt.title('Feature Scaling: radius_mean')
plt.show()


## Confusion Matrix
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


                            # Split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X.shape)
print(X_train.shape)
print(X_test.shape)


                            # Standardize the data using Standard sclaer
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

type(X_train)


                        # Convert data to PyTorch tensors and move it to GPU
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


                                    # Neural Network Architecture
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



                                            # Hyperparameters
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 100

## Initialize the Neural Network and move it the GPU
model = NeuralNet(input_size, hidden_size, output_size).to(device)

## Loss and the Optiizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


                                     # Training the Neural Network
for epoch in range(num_epochs):
  model.train()
  optimizer.zero_grad()
  outputs = model(X_train)
  loss = criterion(outputs, y_train.view(-1,1))
  loss.backward()
  optimizer.step()

## Claculate Accuracy
  with torch.no_grad():
    predicted = outputs.round()
    correct = (predicted == y_train.view(-1,1)).float().sum()
    accuracy = correct/y_train.size(0)

  if (epoch+1) % 10 == 0:
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss : {loss.item():.4f}, Accuracy: {accuracy.item() * 100:.2f}%")


                                        # Model Evaluation
## evaluation on training set
model.eval()
with torch.no_grad():
  outputs = model(X_train)
  predicted = outputs.round()
  correct = (predicted == y_train.view(-1,1)).float().sum()
  accuracy = correct/y_train.size(0)
  print(f"Accuracy on training data: {accuracy.item() * 100:.2f}%")


## evaluation on test set
model.eval()
with torch.no_grad():
  outputs = model(X_test)
  predicted = outputs.round()
  correct = (predicted == y_test.view(-1,1)).float().sum()
  accuracy = correct/y_test.size(0)
  print(f"Accuracy on test data: {accuracy.item() * 100:.2f}%")











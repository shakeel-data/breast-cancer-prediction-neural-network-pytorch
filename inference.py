import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Dummy model class (replace with your real one)
class BreastCancerNN(torch.nn.Module):
    def __init__(self):
        super(BreastCancerNN, self).__init__()
        self.fc1 = torch.nn.Linear(30, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

# Load model
model = BreastCancerNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Load data
df = pd.read_csv("Clinical_data.csv")  # change this if your dataset has a different name
X = df.drop(columns=["diagnosis"], errors='ignore')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Predict
with torch.no_grad():
    predictions = model(X_tensor)
    predicted_classes = (predictions > 0.5).int()

print(predicted_classes)
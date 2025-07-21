```python
import torch
from model import BreastCancerNN  # assuming this exists
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load trained model
model = BreastCancerNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Load new sample data
df = pd.read_csv("breast_cancer_dataset.csv")
X = df.drop("target", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Predict
with torch.no_grad():
    predictions = model(X_tensor)
    predicted_classes = (predictions > 0.5).int()

print(predicted_classes)
```

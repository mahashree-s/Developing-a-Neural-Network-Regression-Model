# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY

Neural network regression is a supervised learning technique used to predict continuous numerical values. In this experiment, a dataset containing one input and one output is created. The dataset is split into training and testing sets, and Min-Max scaling is applied to normalize the data.

A neural network with one input layer, one hidden layer, and one output layer is built using PyTorch. The model is trained using Mean Squared Error (MSE) loss and the Adam optimizer. The training performance is plotted, the model is evaluated using testing data, and a prediction is made for a new input value.

<img width="1116" height="757" alt="image" src="https://github.com/user-attachments/assets/d1540973-a037-42f8-8c9f-32881a19c163" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:MAHASHREE S

### Register Number:212225230163

```python

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.DataFrame({"Input": np.arange(1, 21, dtype=float)})
df["Output"] = 2 * df["Input"] + 3
print(df)
```

```python
X = df[["Input"]].values
y = df[["Output"]].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
```

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ai_brain.parameters(), lr=0.01)

```

```python
def train_model(ai_brain, X_train, y_train, criterion,
                optimizer, epochs=2000):
    losses = []

    for epoch in range(epochs):
        output = ai_brain(X_train)
        loss = criterion(output, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

losses = train_model(
    ai_brain, X_train, y_train, criterion, optimizer
)

print("Training completed")
```

```python
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.title("Training Loss Vs Iteration")
plt.grid()
plt.show()

```

```python
ai_brain.eval()

with torch.no_grad():
    prediction = ai_brain(X_test)
    test_loss = criterion(prediction, y_test).item()

print("Test MSE:", test_loss)
```

```python
new_input = np.array([[25.0]])
scaled_input = scaler_X.transform(new_input)

with torch.no_grad():
    x = torch.tensor(scaled_input, dtype=torch.float32)
    result = ai_brain(x).numpy()

result = scaler_y.inverse_transform(result)

print("New Input:", 25)
print("Predicted Output:", result[0][0])
```

### Dataset Information

<img width="833" height="533" alt="image" src="https://github.com/user-attachments/assets/52df3950-40ef-4e78-8863-a355579d36b6" />

### OUTPUT
<img width="873" height="369" alt="image" src="https://github.com/user-attachments/assets/8b1feb16-46db-4c6d-a34c-a95e4e693792" />

### Training Loss Vs Iteration Plot
The training loss should generally decrease as the model learns.

<img width="784" height="519" alt="image" src="https://github.com/user-attachments/assets/674b364d-4ccc-4781-9c59-b4c98a11c854" />


### New Sample Data Prediction
<img width="546" height="217" alt="image" src="https://github.com/user-attachments/assets/65349691-9204-4f44-9d59-ffbcaf914e2e" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch. The model was evaluated using testing data, and a new numerical output was predicted using the trained model.

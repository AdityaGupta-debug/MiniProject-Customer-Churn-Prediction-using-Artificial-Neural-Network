# 🔍 Customer Churn Prediction using Artificial Neural Networks (ANN)

## 📊 About the Dataset

The dataset used in this project comes from a **telecom company** and contains information about customers, including their demographics, account details, and service usage. The goal is to predict whether a customer will **churn** (i.e., leave the service) or not.

### 🧾 Key Features

| Feature            | Description                                                  |
|--------------------|--------------------------------------------------------------|
| `CreditScore`      | Customer credit score                                        |
| `Geography`        | Customer location (e.g., France, Germany, Spain)             |
| `Gender`           | Male/Female                                                  |
| `Age`              | Age of the customer                                          |
| `Tenure`           | Number of years the customer has stayed with the company     |
| `Balance`          | Account balance                                              |
| `NumOfProducts`    | Number of bank products the customer is using                |
| `HasCrCard`        | Whether the customer has a credit card (1 or 0)              |
| `IsActiveMember`   | Whether the customer is active (1 or 0)                      |
| `EstimatedSalary`  | Estimated salary of the customer                             |
| `Exited`           | Target variable (1 if customer churned, 0 otherwise)         |

---

## 📈 Exploratory Data Analysis (EDA) and Feature Engineering

- Performed univariate and bivariate analysis to understand the distribution and relationships between features.
- Visualized patterns using histograms, countplots, boxplots, and heatmaps.
- Handled categorical data with **Label Encoding** and **One-Hot Encoding**.
- Scaled numerical features using **StandardScaler** for neural network input.

---

## 🧠 Model - Artificial Neural Network (ANN)

An artificial neural network was built using **TensorFlow/Keras** with the following architecture:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(11,)))
model.add(Dense(1, activation='sigmoid'))
```

✅ The model achieved an **accuracy of 86%** on the test dataset.  
📊 Additionally, training and validation performance was tracked using **loss and accuracy graphs** to monitor the model's learning behavior and evaluate for **overfitting or underfitting**.

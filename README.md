# 🔍 Customer Churn Prediction using Artificial Neural Networks (ANN)

## 📊 About the Dataset

The dataset used in this project comes from a **telecom company** and contains information about customers, including their demographics, account details, and service usage. The goal is to predict whether a customer will **churn** (i.e., leave the service) or not.

### 🧾 Key Features:
- `CreditScore`: Customer credit score
- `Geography`: Customer location
- `Gender`: Male/Female
- `Age`: Age of the customer
- `Tenure`: Number of years the customer has stayed with the company
- `Balance`: Account balance
- `NumOfProducts`: Number of bank products the customer is using
- `HasCrCard`: Whether the customer has a credit card
- `IsActiveMember`: Whether the customer is active
- `EstimatedSalary`: Estimated salary of the customer
- `Exited`: Target variable (1 if customer churned, 0 otherwise)

---

## 📈 Exploratory Data Analysis (EDA) and Feature Engineering

- Performed univariate and bivariate analysis to understand the distribution and relationship of features.
- Created visualizations such as histograms, countplots, and boxplots to explore data patterns.
- Applied encoding techniques like **Label Encoding** and **One-Hot Encoding**.
- Scaled numerical features using **StandardScaler** to prepare data for training.

---

## 🧠 Model - Artificial Neural Network (ANN)

- Built an ANN using TensorFlow/Keras with multiple Dense layers.
- Used `ReLU` activation for hidden layers and `Sigmoid` for the output layer.
- Applied binary cross-entropy as the loss function and `Adam` optimizer.

### ✅ Results
- **Accuracy Achieved:** `86%` on the test set.

---

## 📉 Training Performance

The model was evaluated using training and validation loss and accuracy curves:

### 📊 Loss Curve:
![Loss Curve](images/loss_curve.png)

### 📈 Accuracy Curve:
![Accuracy Curve](images/accuracy_curve.png)

---

## 🚀 Conclusion

This project demonstrates the power of deep learning in solving classification problems like customer churn. With effective preprocessing, EDA, and a well-structured ANN, we achieved strong performance in predicting customer churn.

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy, Matplotlib, Seaborn
- TensorFlow / Keras
- Scikit-learn


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Input data (Monthly Income, Monthly Expenses, Home Owner)
X = np.array([[2, 1, 0], 
              [1, 2, 0], 
              [6, 2, 1], 
              [3, 1, 1], 
              [3, 2, 0]])

# Output data (Credit Scores)
y = np.array([3, 1, 5, 4, 2])

# Normalizing the input data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_scaled, y)

# Make predictions on the training data
y_pred = model.predict(X_scaled)

# Calculate the mean squared error
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot the true vs predicted values
plt.scatter(range(len(y)), y, label='True Values', color='blue')
plt.scatter(range(len(y)), y_pred, label='Predicted Values', color='red')
plt.title('True vs Predicted Credit Scores')
plt.xlabel('Sample Index')
plt.ylabel('Credit Score')
plt.legend()
plt.show()

# New customer data (monthly income, monthly expenses, home owner)
new_customer = np.array([[5, 3, 1]])  # Input values for the new customer
new_customer_scaled = scaler.transform(new_customer)

# Predict the credit score for the new customer
predicted_score = model.predict(new_customer_scaled)
print(f"Predicted Credit Score for the new customer: {predicted_score[0]:.2f}")

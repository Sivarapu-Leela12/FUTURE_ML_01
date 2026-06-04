import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
data = pd.read_csv("sales.csv")

# Convert Date column
data['Date'] = pd.to_datetime(data['Date'])

# Create time features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Features and target
X = data[['Year', 'Month', 'Day']]
y = data['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict sales
predictions = model.predict(X)

# Error calculation
mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)

# Future prediction
future = pd.DataFrame({
    'Year': [2024, 2024],
    'Month': [2, 2],
    'Day': [1, 5]
})

future_sales = model.predict(future)

print("\nFuture Sales Prediction:")
print(future_sales)

# Plot graph
plt.figure(figsize=(10,5))

plt.plot(data['Date'], y, label='Actual Sales')
plt.plot(data['Date'], predictions, label='Predicted Sales')

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecasting")

plt.legend()

plt.savefig("forecast_output.png")

plt.show()
print("Forecast running")
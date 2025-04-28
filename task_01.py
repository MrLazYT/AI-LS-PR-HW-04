import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.linspace(-20, 20, 100).reshape(-1, 1)
y = (np.sin(X) + 0.1 * X**2).flatten() + np.random.normal(0, 0.5, size=X.shape[0])

print("X:", X)
print("Y:", y)

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

X_test = np.linspace(-20, 20, 100).reshape(-1, 1)
X_poly = poly.transform(X_test)
y_pred = model.predict(X_poly)

plt.plot(X, y, color='green', label='Predicted Function')
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X_test, y_pred, color='red', label='Real function')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()

x_value = 7
x_value_poly = poly.transform(np.array([[x_value]]))  # Ось це додали
predicted = model.predict(x_value_poly)
print(f"Predicted value at X = {x_value}: {predicted[0]:.2f}")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y, model.predict(X_poly))
mse = mean_squared_error(y, model.predict(X_poly))
r2 = r2_score(y, model.predict(X_poly))

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
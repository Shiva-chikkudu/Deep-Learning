import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x_train = np.random.randn(1500, 20)
y_train = np.random.randn(3000, 1200)

x = np.random.randn(1500, 20).astype("float32")
y = np.random.randn(1500, 1200).astype("float32")

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
)

model = keras.Sequential([
    keras.layers.Input(shape=(20,)),
    keras.layers.Dense(800, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(400, activation="relu"),
    keras.layers.Dense(200, activation="relu"),
    keras.layers.Dense(1200),
])

model.summary()

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
)

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
)

model.summary()
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
)

test_loss, test_mae = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

predictions = model.predict(x_test[:5])
print("Prediction shape:", predictions.shape)
print(
    "Mean squared error:",
    mean_squared_error(y_test[:5], predictions),
)
x = np.random.randn(1500, 20).astype("float32")
y = np.random.randn(1500, 1200).astype("float32")

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model.summary()
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

model = keras.Sequential([
    keras.layers.Input(shape=(20,)),
    keras.layers.Dense(800, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(400, activation="relu"),
    keras.layers.Dense(200, activation="relu"),
    keras.layers.Dense(1200),
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
)

test_loss, test_mae = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

predictions = model.predict(x_test[:5])
print("Prediction shape:", predictions.shape)
print("Mean squared error:", mean_squared_error(y_test[:5], predictions))
# 7. Evaluate on unseen test data
test_loss, test_mae = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")
# Evaluate regression performance
test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Make predictions and compare with the continuous targets
predictions = model.predict(x_test[:5], verbose=0)
print("Prediction shape:", predictions.shape)
print("Mean squared error:", mean_squared_error(y_test[:5], predictions))
print("Confusion matrix is not applicable to this regression model.")
# 8. Make predictions
predictions = model.predict(x_test[:5])
predicted_labels = predictions.argmax(axis=1)

print("Predicted labels:", predicted_labels)
print("Actual labels:   ", y_test[:5], y_train[:5])

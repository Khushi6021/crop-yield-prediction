# ---------------------------
# IMPORT LIBRARIES
# ---------------------------
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

import matplotlib.pyplot as plt
import joblib

# ---------------------------
# CREATE BACKEND FOLDER IF NOT EXISTS
# ---------------------------
if not os.path.exists("backend"):
    os.makedirs("backend")

# ---------------------------
# LOAD DATA
# ---------------------------
print("Code is running...")
df = pd.read_excel("../data.csv.xlsx")

print("Dataset Preview:\n", df.head())

# ---------------------------
# FEATURES & TARGET
# ---------------------------
X = df.drop("Yield", axis=1)
y = df["Yield"]

# ---------------------------
# TRAIN TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# RANDOM FOREST MODEL
# ---------------------------
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

# ---------------------------
# LSTM MODEL
# ---------------------------
X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

lstm_model = Sequential([
    Input(shape=(1, X_train.shape[1])),
    LSTM(50, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.fit(X_train_lstm, y_train, epochs=100, verbose=0)

lstm_pred = lstm_model.predict(X_test_lstm).flatten()

# ---------------------------
# HYBRID MODEL
# ---------------------------
hybrid_pred = (rf_pred + lstm_pred) / 2

# ---------------------------
# EVALUATION
# ---------------------------
def evaluate(y_true, y_pred, model_name):
    print(f"\n{model_name} Performance:")
    print("R2 Score:", r2_score(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))

evaluate(y_test, rf_pred, "Random Forest")
evaluate(y_test, lstm_pred, "LSTM")
evaluate(y_test, hybrid_pred, "Hybrid Model")

# ---------------------------
# SAVE MODELS (VERY IMPORTANT)
# ---------------------------
joblib.dump(rf, "rf_model.pkl")
lstm_model.save("lstm_model.h5")
print("\n✅ Models saved successfully in backend folder!")

# ---------------------------
# GRAPH
# ---------------------------
plt.figure()

plt.plot(y_test.values, marker='o', label="Actual")
plt.plot(rf_pred, marker='x', label="Random Forest")
plt.plot(lstm_pred, marker='s', label="LSTM")
plt.plot(hybrid_pred, marker='^', label="Hybrid")

plt.title("Hybrid Model vs Actual Yield")
plt.xlabel("Test Samples")
plt.ylabel("Yield")
plt.legend()
plt.grid()

plt.show()
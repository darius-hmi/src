import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yf

# Parameters
stock_symbol = input("Please input the code for the stock: ")
start_date = '2011-01-01'
end_date = '2023-12-31'
look_back = 25  # Number of days to look back for predictions

# Download historical stock data
data = yf.download(stock_symbol, start=start_date, end=end_date)
data['Date'] = data.index
data['Price_Change'] = data['Close'].pct_change().shift(-1)
data.dropna(inplace=True)

# Create target variable: 1 if price goes up, 0 if down
data['Target'] = (data['Price_Change'] > 0).astype(int)

# Features and target
features = data[['Close']].values
target = data['Target'].values

# Scale features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Create sequences
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(target[i + look_back])
    return np.array(X), np.array(y)

X, y = create_sequences(features, look_back)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1), return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Predict next day
last_sequence = features[-look_back:].reshape(1, look_back, 1)
prediction = model.predict(last_sequence)
print(f'Prediction for next day: {"Up" if prediction > 0.5 else "Down"}')

import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
from tensorflow.keras import regularizers

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'nn_model.joblib')
scaler_path = os.path.join(current_dir, 'scaler.joblib')
processed_data_path = os.path.join(current_dir, 'processed_data.csv')
label_encoder_path = os.path.join(current_dir, 'label_encoder.joblib')

sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import prepare_data_for_training, prepare_data_for_training_binary

data = pd.read_csv('../../data/week13.csv')
X, y, df, label_encoder = prepare_data_for_training(data)

df.to_csv(processed_data_path, index=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),  # Input layer
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1000, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.02)),            # Hidden layer 1
    tf.keras.layers.Dense(400, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'), 
    tf.keras.layers.Dense(3, activation='softmax')           # Output layer (3 classes: win, draw, lose)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.007)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=1)

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoder, label_encoder_path)


print(f"Test Accuracy: {test_accuracy:.2f}")


y_train_pred_probs = model.predict(X_train_scaled)
y_train_pred = tf.argmax(y_train_pred_probs, axis=1).numpy()

y_test_pred_probs = model.predict(X_test_scaled)
y_test_pred = tf.argmax(y_test_pred_probs, axis=1).numpy()

# Training data evaluation
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Neural Network Model - Training Accuracy: {train_accuracy:.2f}")
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

# Test data evaluation
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Neural Network Model - Test Accuracy: {test_accuracy:.2f}")
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))


# import matplotlib.pyplot as plt

# pd.DataFrame(history.history).plot(
#     figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",
#     style=["r--", "r--.", "b-", "b-*"])
# plt.legend(loc="lower left")  # extra code
# plt.show()

#this cant be right

# Training Classification Report:
#               precision    recall  f1-score   support

#            0       0.88      0.87      0.87       891
#            1       0.84      0.85      0.84       717

#     accuracy                           0.86      1608
#    macro avg       0.86      0.86      0.86      1608
# weighted avg       0.86      0.86      0.86      1608

# Neural Network Model - Test Accuracy: 0.86
# Test Classification Report:
#               precision    recall  f1-score   support

#            0       0.90      0.85      0.87       232
#            1       0.81      0.86      0.84       170

#     accuracy                           0.86       402
#    macro avg       0.85      0.86      0.85       402
# weighted avg       0.86      0.86      0.86       402


#goodenough
# Training Classification Report:
#               precision    recall  f1-score   support

#            0       0.86      0.87      0.87       891
#            1       0.84      0.83      0.83       717

#     accuracy                           0.85      1608
#    macro avg       0.85      0.85      0.85      1608
# weighted avg       0.85      0.85      0.85      1608

# Neural Network Model - Test Accuracy: 0.85
# Test Classification Report:
#               precision    recall  f1-score   support

#            0       0.88      0.86      0.87       232
#            1       0.81      0.84      0.82       170

#     accuracy                           0.85       402
#    macro avg       0.84      0.85      0.85       402
# weighted avg       0.85      0.85      0.85       402
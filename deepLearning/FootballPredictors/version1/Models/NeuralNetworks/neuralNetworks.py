import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'nn_model.joblib')
scaler_path = os.path.join(current_dir, 'scaler.joblib')
processed_data_path = os.path.join(current_dir, 'processed_data.csv')
label_encoder_path = os.path.join(current_dir, 'label_encoder.joblib')

sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import prepare_data_for_training, prepare_data_for_training_binary

data = pd.read_csv('../../data/week14.csv')
#data = data.drop(columns=['Home_Form', 'Away_Form', 'Home_Form2', 'Away_Form2', 'Home_Goals_Last_3', 'Away_Goals_Last_3', 'Home_Goals_Conceded_Last_3', 'Away_Goals_Conceded_Last_3'])
X, y, df, label_encoder = prepare_data_for_training(data)

df.to_csv(processed_data_path, index=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# def lr_schedule(epoch, lr):
#     if epoch < 10:
#         return lr  # Keep the initial learning rate for the first 10 epochs
#     else:
#         return lr * 0.9  # Reduce the learning rate by a factor of 0.9 every epoch after 10

early_stopping = EarlyStopping(monitor='val_loss',  # Monitor validation loss
                               patience=10,         # Stop after 10 epochs without improvement
                               restore_best_weights=True,  # Restore the best weights after stopping
                               verbose=1) 

#lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
#lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.5,
    patience=3,  # Try lowering patience
    min_lr=1e-6,  # Lower the minimum learning rate
    verbose=1
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),  # Input layer



    # tf.keras.layers.Dense(800, kernel_initializer="he_normal", use_bias=True),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Activation("relu"),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(800, kernel_initializer="he_normal", use_bias=True),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Activation("relu"),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(800, kernel_initializer="he_normal", use_bias=True),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Activation("relu"),
    # tf.keras.layers.Dropout(0.2),


    tf.keras.layers.Dense(100, activation='relu', kernel_initializer="he_normal", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer="he_normal", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer="he_normal", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
    
])

# if using adamw then remove L2 reg
# optimizer = tf.keras.optimizers.AdamW(weight_decay=1e-5, learning_rate=0.001,
#                                       beta_1=0.9, beta_2=0.999)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9,
                                    nesterov=True)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



history = model.fit(X_train_scaled, y_train, 
                    epochs=100, 
                    validation_split=0.2, # Set the maximum number of epochs you want to run
                    callbacks=[early_stopping, lr_scheduler])  # Add EarlyStopping callback

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


# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Confusion matrix for training set
# train_cm = confusion_matrix(y_train, y_train_pred)
# test_cm = confusion_matrix(y_test, y_test_pred)

# # Plot confusion matrix
# def plot_confusion_matrix(cm, title):
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
#                 yticklabels=label_encoder.classes_)
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title(title)
#     plt.show()

# plot_confusion_matrix(train_cm, "Training Set Confusion Matrix")
# plot_confusion_matrix(test_cm, "Test Set Confusion Matrix")

# # Convert predictions to probabilities
# y_test_pred_probs = model.predict(X_test_scaled)

# # Plot probability distribution for each class
# for i, class_name in enumerate(label_encoder.classes_):
#     plt.figure(figsize=(6, 4))
#     sns.histplot(y_test_pred_probs[:, i], bins=20, kde=True, color=f"C{i}", label=class_name)
#     plt.title(f"Probability Distribution for Class '{class_name}'")
#     plt.xlabel("Predicted Probability")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.show()

# import matplotlib.pyplot as plt

# pd.DataFrame(history.history).plot(
#     figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",
#     style=["r--", "r--.", "b-", "b-*"])
# plt.legend(loc="lower left")  # extra code
# plt.show()

#this cant be right
#below is accuracy of current with 3 outcomes 
# Neural Network Model - Training Accuracy: 0.78
# Training Classification Report:
#               precision    recall  f1-score   support

#            0       0.63      0.47      0.54       512
#            1       0.83      0.89      0.86       996
#            2       0.80      0.86      0.83       716

#     accuracy                           0.78      2224
#    macro avg       0.75      0.74      0.74      2224
# weighted avg       0.77      0.78      0.78      2224

# Neural Network Model - Test Accuracy: 0.72
# Test Classification Report:
#               precision    recall  f1-score   support

#            0       0.45      0.38      0.41       123
#            1       0.80      0.82      0.81       246
#            2       0.78      0.82      0.80       187

#     accuracy                           0.72       556
#    macro avg       0.67      0.68      0.67       556
# weighted avg       0.71      0.72      0.72       556
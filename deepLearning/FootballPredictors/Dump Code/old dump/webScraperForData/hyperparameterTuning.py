

#Copied over from ssome legacy code to be studied


from main_RF_features import prepare_data  # Import the prepare_data function
from sklearn.model_selection import GridSearchCV, train_test_split
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# Load and prepare data
features_scaled, target, label_encoder_home, label_encoder_away, scaler, home_stats, away_stats = prepare_data('data.csv', decay_factor=0.1)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Define the model creation function
def create_model(neurons=64, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(int(neurons / 2), activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Wrap the model with KerasClassifier
model = KerasClassifier(model=create_model, verbose=0)

# Define the grid of hyperparameters
param_grid = {
    'model__neurons': [32, 64, 128],  # Use 'model__' prefix for hyperparameters of the model
    'model__learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'epochs': [10, 20, 30]
}

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# Print and save the best hyperparameters
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(file_path, target_column):
    """Load and prepare the dataset."""
    # Load the dataset
    data = pd.read_csv(file_path)

    # Encode categorical features
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        if column != target_column:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le

    # Split data into features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X, y, label_encoders

def find_feature_importance(X, y):
    """Train a Random Forest model and plot feature importance."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Get feature importance
    feature_importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return feature_importance_df

def plot_feature_importance(feature_importance_df, top_n=20):
    """Plot the top N most important features."""
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n))
    plt.title(f'Top {top_n} Most Important Features')
    plt.show()

if __name__ == "__main__":
    # Define the file path and target column
    file_path = 'cleaned_file_noNull_no_noNonInteger.csv'
    target_column = 'Result'  # Change this to the actual name of your target column

    # Load and prepare data
    X, y, label_encoders = load_and_prepare_data(file_path, target_column)

    # Find feature importance
    feature_importance_df = find_feature_importance(X, y)

    # Display feature importance
    print(feature_importance_df)

    # Plot the top N most important features
    plot_feature_importance(feature_importance_df, top_n=30)

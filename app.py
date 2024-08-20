from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# Load the model and data preprocessing setup
file_path = 'student_career_data.csv'  # Update this path as needed

# Load the CSV file
data = pd.read_csv(file_path)

# Ensure the label encoder is fitted with all possible values, including "Unknown"
label_encoder = LabelEncoder()
all_possible_values = list(data['career_interests'].unique()) + ['Unknown']
label_encoder.fit(all_possible_values)

# Convert categorical data to numerical
data['preferred_work_environment'] = data['preferred_work_environment'].map(lambda s: s if s in label_encoder.classes_ else 'Unknown')
data['preferred_work_environment'] = label_encoder.transform(data['preferred_work_environment'])
data['career_interests'] = label_encoder.transform(data['career_interests'])

# Function to preprocess aptitude test scores
def preprocess_aptitude_scores(scores):
    # Handle both comma and semicolon separators
    if ',' in scores:
        separator = ','
    elif ';' in scores:
        separator = ';'
    else:
        raise ValueError("Aptitude test scores must be separated by either ',' or ';'")
    return np.mean([int(score) for score in scores.split(separator)])

# Apply preprocessing to aptitude test scores
data['aptitude_test_scores'] = data['aptitude_test_scores'].apply(preprocess_aptitude_scores)

# Normalize the aptitude test scores
scaler = StandardScaler()
data['aptitude_test_scores'] = scaler.fit_transform(data['aptitude_test_scores'].values.reshape(-1, 1))

# Function to one-hot encode multi-category columns
def one_hot_encode_multi_category_column(column):
    mlb = MultiLabelBinarizer()
    categories = mlb.fit_transform(column.str.split(';'))
    return pd.DataFrame(categories, columns=mlb.classes_)

# One-hot encode the multi-category features
favorite_subjects_encoded = one_hot_encode_multi_category_column(data['favorite_subjects'])
hobbies_encoded = one_hot_encode_multi_category_column(data['hobbies'])
strengths_encoded = one_hot_encode_multi_category_column(data['strengths'])
weaknesses_encoded = one_hot_encode_multi_category_column(data['weaknesses'])

# Concatenate the encoded features back to the main dataframe
data = pd.concat([data, favorite_subjects_encoded, hobbies_encoded, strengths_encoded, weaknesses_encoded], axis=1)

# Drop the original columns
data.drop(['username', 'favorite_subjects', 'hobbies', 'strengths', 'weaknesses'], axis=1, inplace=True)

# Define features and target
X = data.drop(['career_interests'], axis=1)
y = data['career_interests']

# Remove classes with very few samples (e.g., less than 6)
min_samples = 2  # Set a lower threshold to ensure sufficient samples
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= min_samples].index
valid_indices = y.isin(valid_classes)
X = X[valid_indices]
y = y[valid_indices]

# Balance the dataset using SMOTE
min_samples_after_filtering = min(y.value_counts())
k_neighbors_value = min(min_samples_after_filtering - 1, 5)
smote = SMOTE(random_state=42, k_neighbors=k_neighbors_value)
X_res, y_res = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize the RandomForestClassifier
clf = RandomForestClassifier(random_state=42)

# Perform grid search to find the best parameters
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters and retrain the model
best_params = grid_search.best_params_
clf = RandomForestClassifier(**best_params, random_state=42)
clf.fit(X_train, y_train)

# Calculate the accuracy on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100  # Convert to percentage

# Function to predict career interests for new data
def predict_career(new_data):
    # Preprocess the new data
    new_data['preferred_work_environment'] = new_data['preferred_work_environment'].map(lambda s: s if s in label_encoder.classes_ else 'Unknown')
    new_data['preferred_work_environment'] = label_encoder.transform(new_data['preferred_work_environment'])
    new_data['aptitude_test_scores'] = new_data['aptitude_test_scores'].apply(preprocess_aptitude_scores)
    new_data['aptitude_test_scores'] = scaler.transform(np.array(new_data['aptitude_test_scores']).reshape(-1, 1))

    # One-hot encode the multi-category features for new data
    favorite_subjects_encoded = one_hot_encode_multi_category_column(new_data['favorite_subjects'])
    hobbies_encoded = one_hot_encode_multi_category_column(new_data['hobbies'])
    strengths_encoded = one_hot_encode_multi_category_column(new_data['strengths'])
    weaknesses_encoded = one_hot_encode_multi_category_column(new_data['weaknesses'])

    # Concatenate the encoded features back to the main dataframe
    new_data = pd.concat([new_data, favorite_subjects_encoded, hobbies_encoded, strengths_encoded, weaknesses_encoded], axis=1)

    # Drop the original columns
    new_data.drop(['username', 'favorite_subjects', 'hobbies', 'strengths', 'weaknesses'], axis=1, inplace=True)

    # Ensure all columns match the training set
    missing_cols = set(X.columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0
    new_data = new_data[X.columns]

    # Make a prediction
    prediction = clf.predict(new_data)
    prediction_label = label_encoder.inverse_transform(prediction)

    return prediction_label[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
        'username': request.form['username'],
        'age': int(request.form['age']),
        'grade': int(request.form['grade']),
        'favorite_subjects': request.form['favorite_subjects'],
        'hobbies': request.form['hobbies'],
        'preferred_work_environment': request.form['preferred_work_environment'],
        'strengths': request.form['strengths'],
        'weaknesses': request.form['weaknesses'],
        'aptitude_test_scores': request.form['aptitude_test_scores']
    }
    
    new_data = pd.DataFrame([data])
    
    # Predict the career interest
    prediction = predict_career(new_data)
    
    # Render the result template with the prediction and accuracy
    return render_template('result.html', career_interest=prediction, accuracy=round(accuracy, 2))

if __name__ == '__main__':
    app.run(debug=True)

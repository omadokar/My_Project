import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from imblearn.over_sampling import SMOTE

# Define the path to your CSV file
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
    return np.mean([int(score) for score in scores.split(';')])

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

# Drop any rows with NaN values
X = X.dropna()
y = y[X.index]

# Print the class distribution before filtering
print("Class distribution before filtering:")
print(y.value_counts())

# Remove classes with very few samples (e.g., less than 6)
min_samples = 2  # Set a lower threshold to ensure sufficient samples
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= min_samples].index
valid_indices = y.isin(valid_classes)
X = X[valid_indices]
y = y[valid_indices]

# Print the class distribution after filtering
print("Class distribution after filtering:")
print(y.value_counts())

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)
X_res, y_res = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print('Classification Report:')
print(classification_report(y_test, y_pred, zero_division=0))
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

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

# Example usage with new data for MBBS and Pharmacy
new_student_data_1 = {
    "username": "student_1",
    "age": 18,
    "grade": 12,
    "favorite_subjects": "Math;Computer Science",
    "hobbies": "Reading;Gaming",
    "preferred_work_environment": "Indoor",
    "strengths": "Logical thinking;Programming",
    "weaknesses": "Public speaking",
    "aptitude_test_scores": "85;90;80"
}

new_student_data_2 = {
    "username": "student_2",
    "age": 17,
    "grade": 11,
    "favorite_subjects": "Biology;Chemistry",
    "hobbies": "Painting;Music",
    "preferred_work_environment": "Outdoor",
    "strengths": "Creativity;Empathy",
    "weaknesses": "Time management",
    "aptitude_test_scores": "75;80;85"
}

new_student_data_3 = {
    "username": "student_3",
    "age": 18,
    "grade": 12,
    "favorite_subjects": "Biology;Chemistry",
    "hobbies": "Writing;Reading",
    "preferred_work_environment": "Indoor",
    "strengths": "Empathy;Critical thinking",
    "weaknesses": "Public speaking",
    "aptitude_test_scores": "90;85;88"
}

new_student_df_1 = pd.DataFrame([new_student_data_1])
new_student_df_2 = pd.DataFrame([new_student_data_2])
new_student_df_3 = pd.DataFrame([new_student_data_3])

predicted_career_1 = predict_career(new_student_df_1)
predicted_career_2 = predict_career(new_student_df_2)
predicted_career_3 = predict_career(new_student_df_3)

print(f'Predicted Career Interest for student 1: {predicted_career_1}')
print(f'Predicted Career Interest for student 2: {predicted_career_2}')
print(f'Predicted Career Interest for student 3: {predicted_career_3}')
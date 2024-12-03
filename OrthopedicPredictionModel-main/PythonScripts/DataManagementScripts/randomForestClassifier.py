import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import numpy as np

def load_data():
    patients = pd.read_csv('../data/patients_cleaned.csv')
    conditions = pd.read_csv('../data/conditions_cleaned.csv')
    procedures = pd.read_csv('../data/procedures_cleaned.csv')
    implants = pd.read_csv('../data/implants_cleaned.csv')
    return patients, conditions, procedures, implants

def preprocess_data(patients, conditions):
    # Merging data
    merged_data = pd.merge(patients, conditions, on='patient_id')
    merged_data.dropna(inplace=True)

    # Encoding categorical features
    le_gender = LabelEncoder()
    merged_data['gender'] = le_gender.fit_transform(merged_data['gender'])

    le_condition = LabelEncoder()
    merged_data['condition_name'] = le_condition.fit_transform(merged_data['condition_name'])

    le_severity = LabelEncoder()
    merged_data['severity'] = le_severity.fit_transform(merged_data['severity'])

    # Features and target variable
    features = merged_data[['age', 'gender', 'severity']]
    target = merged_data['condition_name']

    # Scaling the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target, le_condition, scaler

def tune_and_train_model(features, target):
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True]
    }

    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        n_jobs=-1,
        scoring='accuracy',
        random_state=42
    )
    random_search.fit(x_train, y_train)

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(x_test)

    # Saving the best model
    joblib.dump(best_model, '../data/random_forest_model.pkl')
    print("Model saved successfully.")
    return best_model, y_pred, y_test, random_search.best_params_

def evaluate_model(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

def recommend_treatment(input_data, le_condition, scaler, procedures, implants):
    # Load the saved model
    model = joblib.load('../data/random_forest_model.pkl')

    # Scale the input data
    input_data_scaled = scaler.transform([input_data])

    # Predict condition
    predicted_condition_encoded = model.predict(input_data_scaled)[0]
    predicted_condition = le_condition.inverse_transform([predicted_condition_encoded])[0]

    # Simulated procedure mapping based on the predicted condition
    condition_procedure_map = {
        "Osteoarthritis": ["Knee Arthroscopy", "Knee Replacement", "Physical Therapy"],
        "Rheumatoid Arthritis": ["Joint Replacement", "Medication", "Physical Therapy"],
        "Fracture": ["Fracture Fixation", "Orthopedic Surgery", "Physical Therapy"]
    }

    # Get the relevant procedures for the predicted condition
    relevant_procedures = condition_procedure_map.get(predicted_condition, ["No condition-related procedure mapping available."])

    # Filter compatible implants based on the predicted condition
    compatible_implants = implants[implants['compatibility_conditions'].str.contains(str(predicted_condition), na=False)]
    if compatible_implants.empty:
        implant_recommendation = ["No compatible implants found."]
    else:
        implant_recommendation = compatible_implants['implant_type'].tolist()

    return predicted_condition, relevant_procedures, implant_recommendation

def main():
    # Load and preprocess data
    patients, conditions, procedures, implants = load_data()
    features, target, le_condition, scaler = preprocess_data(patients, conditions)

    # Tune model and train
    model, y_pred, y_test, best_params = tune_and_train_model(features, target)
    print(f"Best Model Parameters: {best_params}")

    # Evaluate model performance
    evaluate_model(y_pred, y_test)

    # Return the label encoder and scaler for prediction later
    return le_condition, scaler, procedures, implants

if __name__ == "__main__":
    le_condition, scaler, procedures, implants = main()

    # Example input data: 50 years old, Female (1), severity=2
    input_data = [50, 1, 2]

    # Get treatment recommendations
    predicted_condition, procedures, implant = recommend_treatment(input_data, le_condition, scaler, procedures, implants)

    # Display results
    print(f"Predicted Condition: {predicted_condition}")
    print(f"Recommended Procedures: {procedures}")
    print(f"Recommended Implant: {implant}")

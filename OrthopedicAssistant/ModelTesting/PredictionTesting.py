import pandas as pd
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load trained models and metadata
logging.info("Loading trained models and metadata...")
implant_model_data = joblib.load("../SavedModels/implant_model.pkl")
procedure_model_data = joblib.load("../SavedModels/procedure_model.pkl")

implant_model = implant_model_data["model"]
procedure_model = procedure_model_data["model"]
label_encoders = implant_model_data["label_encoders"]

# Feature columns order (must match training)
feature_columns = ["Age", "Gender", "BMI", "Activity Level", "Comorbidities",
                   "Smoking Status", "Alcohol Use", "Deformity", "Bone Quality", "Scenario"]


def encode_input(column, value):
    """Encodes categorical values using stored label encoders."""
    encoder = label_encoders.get(column)
    if encoder and value in encoder.classes_:
        return encoder.transform([value])[0]
    return encoder.transform([encoder.classes_[0]])[0] if encoder else value


def predict_implant_procedure(test_cases, expected_outcomes=None):
    """Runs predictions for given test cases and compares with expected outcomes."""
    logging.info("Starting predictions...")
    print(
        f"{'Scenario':<35} {'Expected Implant':<40} {'Expected Procedure':<40} {'Predicted Implant':<40} {'Predicted Procedure':<40}")
    print("=" * 200)

    for idx, test_case in enumerate(test_cases):
        # Encode categorical features
        encoded_inputs = [
            test_case[0],  # Age (numerical)
            encode_input("Gender", test_case[1]),
            test_case[2],  # BMI (numerical)
            encode_input("Activity Level", test_case[3]),
            encode_input("Comorbidities", test_case[4]),
            encode_input("Smoking Status", test_case[5]),
            encode_input("Alcohol Use", test_case[6]),
            encode_input("Deformity", test_case[7]),
            encode_input("Bone Quality", test_case[8]),
            encode_input("Scenario", test_case[9])
        ]

        input_df = pd.DataFrame([encoded_inputs], columns=feature_columns)

        # Predict implant and procedure
        implant_prediction = implant_model.predict(input_df)[0]
        procedure_prediction = procedure_model.predict(input_df)[0]

        # Output results
        if expected_outcomes:
            expected_implant, expected_procedure = expected_outcomes[idx]
            print(
                f"{test_case[9]:<35} {expected_implant:<40} {expected_procedure:<40} {implant_prediction:<40} {procedure_prediction:<40}")
        else:
            print(f"\n--- Prediction Results ---")
            print(f"Scenario: {test_case[9]}")
            print(f"Recommended Implant: {implant_prediction}")
            print(f"Recommended Procedure: {procedure_prediction}")


# Sample Test Cases
test_cases = [
    [65, "Female", 25.3, "High", "Diabetes", "Non-smoker", "Regular", "None", "Normal", "Mechanical Failure"],
    [75, "Male", 28.4, "Moderate", "Osteoporosis", "Former Smoker", "Occasional", "Varus", "Osteoporotic",
     "Osteoarthritis"],
    [50, "Female", 32.1, "Low", "Osteoporosis", "Current Smoker", "Occasional", "Valgus", "Normal",
     "Distal Femoral Fracture"],
    [68, "Male", 26.5, "High", "Diabetes", "Non-smoker", "Regular", "Rotational", "Osteoporotic",
     "Post-Traumatic Arthritis"],
    [82, "Female", 29.8, "Moderate", "Rheumatoid Arthritis", "Former Smoker", "Regular", "None", "Normal",
     "Osteoarthritis"]
]

expected_outcomes = [
    ("Hinged Knee Replacement", "Total Knee Arthroplasty"),  # Mechanical Failure
    ("Total Knee Replacement", "Total Knee Arthroplasty"),  # Osteoarthritis
    ("Distal Femoral Replacement", "ORIF (Open Reduction Internal Fixation)"),  # Distal Femoral Fracture
    ("Total Knee Replacement", "Partial Knee Arthroplasty"),  # Post-Traumatic Arthritis
    ("Total Knee Replacement", "Total Knee Arthroplasty")  # Osteoarthritis
]

# Run Testing
predict_implant_procedure(test_cases, expected_outcomes)
import pandas as pd
import joblib
import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
logging.info("Loading dataset...")
data = pd.read_csv("../SavedDataset/OptimizedDataSet.csv")

# Encode categorical features
categorical_columns = ["Gender", "Activity Level", "Comorbidities", "Smoking Status",
                       "Alcohol Use", "Deformity", "Bone Quality", "Scenario"]

label_encoders = {col: LabelEncoder().fit(data[col]) for col in categorical_columns}
for col, encoder in label_encoders.items():
    data[col] = encoder.transform(data[col])

# Define features and targets
X = data.drop(columns=["Recommended Implant", "Recommended Procedure"])
y_implant = data["Recommended Implant"]
y_procedure = data["Recommended Procedure"]

# Train-test split
X_train, X_test, y_train_implant, y_test_implant, y_train_procedure, y_test_procedure = train_test_split(
    X, y_implant, y_procedure, test_size=0.2, random_state=42
)

# Define hyperparameter search space
param_dist = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5, 10],
    'clf__class_weight': ['balanced', 'balanced_subsample']
}

# Create a pipeline with feature scaling and selection
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("feature_selection", SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")),
    ("clf", RandomForestClassifier(random_state=42))
])

# Use RandomizedSearchCV for hyperparameter tuning
implant_model = RandomizedSearchCV(pipeline, param_distributions=param_dist, cv=3, n_iter=5, scoring='accuracy', n_jobs=-1, random_state=42)
procedure_model = RandomizedSearchCV(pipeline, param_distributions=param_dist, cv=3, n_iter=5, scoring='accuracy', n_jobs=-1, random_state=42)

# Train models
logging.info("Training implant model...")
implant_model.fit(X_train, y_train_implant)
logging.info("Training procedure model...")
procedure_model.fit(X_train, y_train_procedure)

# Make predictions
implant_predictions = implant_model.predict(X_test)
procedure_predictions = procedure_model.predict(X_test)

# Evaluate models
logging.info(f"Implant Model Accuracy: {accuracy_score(y_test_implant, implant_predictions):.4f}")
logging.info(f"Procedure Model Accuracy: {accuracy_score(y_test_procedure, procedure_predictions):.4f}")
logging.info("Implant Model Report:\n" + classification_report(y_test_implant, implant_predictions, zero_division=1))
logging.info("Procedure Model Report:\n" + classification_report(y_test_procedure, procedure_predictions, zero_division=1))

# Save models with metadata
logging.info("Saving models...")
joblib.dump({"model": implant_model.best_estimator_, "version": "1.0", "label_encoders": label_encoders}, "../SavedModels/implant_model.pkl")
joblib.dump({"model": procedure_model.best_estimator_, "version": "1.0", "label_encoders": label_encoders}, "../SavedModels/procedure_model.pkl")

logging.info("Optimized models saved successfully.")
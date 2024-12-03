import pandas as pd


def load_data():
    try:
        patients = pd.read_csv('../data/patients.csv')
        encounters = pd.read_csv('../data/encounters.csv')
        conditions = pd.read_csv('../data/conditions.csv')
        procedures = pd.read_csv('../data/procedures.csv')
        implants = pd.read_csv('../data/implants.csv')
        return patients, encounters, conditions, procedures, implants
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None


def check_missing_values(patients, encounters, conditions, procedures, implants):
    for name, df in zip(['Patients', 'Encounters', 'Conditions', 'Procedures', 'Implants'],
                        [patients, encounters, conditions, procedures, implants]):
        print(f"Missing values in {name}:")
        print(df.isnull().sum())


def preprocess_data(patients, encounters, conditions, procedures, implants):
    check_missing_values(patients, encounters, conditions, procedures, implants)

    # Fill missing values using forward fill
    patients.ffill(inplace=True)
    encounters.ffill(inplace=True)
    conditions.ffill(inplace=True)
    procedures.ffill(inplace=True)
    implants.ffill(inplace=True)

    # Correct data types
    patients['age'] = patients['age'].astype(int)
    encounters['encounter_date'] = pd.to_datetime(encounters['encounter_date'])

    # Ensure 'severity' is consistent in conditions
    conditions['severity'] = conditions['severity'].astype('category')

    # For implants, 'implant_type' and 'manufacturer' are categorical, so ensure they're handled properly
    implants['implant_type'] = implants['implant_type'].astype('category')
    implants['manufacturer'] = implants['manufacturer'].astype('category')

    # If there's any extra category handling for conditions, we can do it here, e.g.:
    conditions['condition_name'] = conditions['condition_name'].astype('category')

    # Handle 'compatibility_conditions' in implants to make sure it’s processed correctly
    implants['compatibility_conditions'] = implants['compatibility_conditions'].apply(
        lambda x: x.split(',') if isinstance(x, str) else x)

    # Ensure consistent date format
    if 'date_diagnosed' in conditions.columns:
        conditions['date_diagnosed'] = pd.to_datetime(conditions['date_diagnosed'], errors='coerce')

    # Additional checks for conditions: map severity levels to a numeric scale for modeling
    severity_mapping = {'mild': 1, 'moderate': 2, 'severe': 3}
    conditions['severity_numeric'] = conditions['severity'].map(severity_mapping)

    # If needed, add more custom preprocessing steps based on specific fields or conditions

    return patients, encounters, conditions, procedures, implants


def main():
    patients, encounters, conditions, procedures, implants = load_data()
    if patients is None:
        return
    patients, encounters, conditions, procedures, implants = preprocess_data(patients, encounters, conditions,
                                                                             procedures, implants)
    # Save cleaned data to new files
    patients.to_csv('../data/patients_cleaned.csv', index=False)
    encounters.to_csv('../data/encounters_cleaned.csv', index=False)
    conditions.to_csv('../data/conditions_cleaned.csv', index=False)
    procedures.to_csv('../data/procedures_cleaned.csv', index=False)
    implants.to_csv('../data/implants_cleaned.csv', index=False)
    print("Data loaded and preprocessed successfully.")


if __name__ == "__main__":
    main()
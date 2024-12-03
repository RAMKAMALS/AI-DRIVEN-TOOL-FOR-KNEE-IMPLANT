import pandas as pd
import numpy as np
import os


def generate_patients(num_patients=1000):
    patients = pd.DataFrame({
        'patient_id': range(1, num_patients + 1),
        'name': [f'Patient_{i}' for i in range(1, num_patients + 1)],
        'age': np.random.randint(20, 80, num_patients),
        'gender': np.random.choice(['Male', 'Female'], num_patients),
        'height_cm': np.random.randint(150, 200, num_patients),
        'weight_kg': np.random.randint(50, 120, num_patients),
    })
    return patients


def generate_encounters(num_patients=1000, num_encounters=3000):
    encounters = pd.DataFrame({
        'encounter_id': range(1, num_encounters + 1),
        'patient_id': np.random.choice(range(1, num_patients + 1), num_encounters),
        'encounter_date': pd.to_datetime(np.random.choice(pd.date_range('2020-01-01', '2023-01-01'), num_encounters)),
        'type': np.random.choice(['check-up', 'surgery', 'follow-up'], num_encounters),
    })
    return encounters


def generate_conditions(patients, num_patients=1000):
    conditions = pd.DataFrame({
        'condition_id': range(1, num_patients + 1),
        'patient_id': patients['patient_id'],
        'condition_name': np.random.choice([
            'Osteoarthritis',
            'Rheumatoid Arthritis',
            'ACL Tear',
            'PCL Tear',
            'Meniscus Tear',
            'Patellofemoral Pain Syndrome',
            'Baker\'s Cyst'], num_patients),
        'severity': np.random.choice(['mild', 'moderate', 'severe'], num_patients),
        'date_diagnosed': pd.to_datetime(np.random.choice(pd.date_range('2015-01-01', '2023-01-01'), num_patients))
    })
    return conditions


def generate_procedures(patients, num_procedures=2000):
    procedure_options = {
        'Osteoarthritis': ['Total Knee Replacement', 'Partial Knee Replacement'],
        'Rheumatoid Arthritis': ['Total Knee Replacement', 'Knee Arthroscopy'],
        'ACL Tear': ['ACL Reconstruction'],
        'PCL Tear': ['PCL Reconstruction'],
        'Meniscus Tear': ['Meniscectomy', 'Meniscus Repair'],
        'Patellofemoral Pain Syndrome': ['Knee Arthroscopy', 'Patellar Realignment'],
        'Baker\'s Cyst': ['Cyst Removal']
    }

    procedures = pd.DataFrame({
        'procedure_id': range(1, num_procedures + 1),
        'patient_id': np.random.choice(patients['patient_id'], num_procedures),
        'procedure_name': [np.random.choice(procedure_options[condition]) for condition in
                           np.random.choice(list(procedure_options.keys()), num_procedures)],
        'success_rate': np.round(np.random.uniform(70, 95, num_procedures), 1)
    })
    return procedures


def generate_implants():
    implants = pd.DataFrame({
        'implant_id': range(1, 6),
        'implant_type': ['Metal-Polyethylene', 'Ceramic-Metal', 'Metal-Metal',
                         'Ceramic-Polyethylene', 'Polyethylene'],
        'manufacturer': ['OrthoCorp', 'BioImplants', 'FlexiJoint', 'KneeMend', 'JointSecure'],
        'average_lifespan': [15, 20, 10, 18, 12],
        'compatibility_conditions': [
            'Osteoarthritis',
            'Meniscus Tear',
            'Rheumatoid Arthritis',
            'ACL Tear',
            'Patellofemoral Pain Syndrome']
    })
    return implants


def main():
    if not os.path.exists('../data'):
        os.makedirs('../data')

    num_patients = 1000
    patients = generate_patients(num_patients)
    patients.to_csv('../data/patients.csv', index=False)

    encounters = generate_encounters(num_patients)
    encounters.to_csv('../data/encounters.csv', index=False)

    conditions = generate_conditions(patients, num_patients)
    conditions.to_csv('../data/conditions.csv', index=False)

    procedures = generate_procedures(patients)
    procedures.to_csv('../data/procedures.csv', index=False)

    implants = generate_implants()
    implants.to_csv('../data/implants.csv', index=False)

    print("All datasets generated and saved successfully.")


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np

np.random.seed(42)

# Defining mappings for implants and procedures based on clinical scenarios
mappings = {
    "implants": {
        "Fractures": ["Distal Femoral Replacement", "Tibial Plateau Prosthesis"],
        "Arthritis": ["Unicompartmental Knee Replacement", "Total Knee Replacement"],
        "Tumors": ["Custom Tumor Prosthesis"],
        "Instability": ["Hinged Knee Replacement", "Patellofemoral Joint Replacement", "Total Knee Replacement"],
        "Cartilage/Ligament": ["Osteochondral Allograft", "Unicompartmental Knee Replacement"]
    },
    "procedures": {
        "Distal Femoral Replacement": ["ORIF (Open Reduction Internal Fixation)", "Bone Grafting"],
        "Tibial Plateau Prosthesis": ["ORIF (Open Reduction Internal Fixation)", "Bone Grafting"],
        "Unicompartmental Knee Replacement": ["Partial Knee Arthroplasty"],
        "Total Knee Replacement": ["Total Knee Arthroplasty", "Partial Knee Arthroplasty"],
        "Hinged Knee Replacement": ["Two-Stage Revision with Spacer", "Total Knee Arthroplasty"],
        "Patellofemoral Joint Replacement": ["Joint Resurfacing", "Patellar Resurfacing"],
        "Osteochondral Allograft": ["Autologous Chondrocyte Implantation", "Arthroscopic Meniscal Repair"],
        "Custom Tumor Prosthesis": ["Bone Grafting", "Wide Tumor Excision"]
    }
}


def generate_synthetic_data(num_samples=10000):
    scenarios = [
        "Distal Femoral Fracture", "Proximal Tibial Fracture", "Patellar Fracture",
        "Osteoarthritis", "Rheumatoid Arthritis", "Post-Traumatic Arthritis",
        "Primary Bone Tumor", "Metastatic Lesion",
        "Mechanical Failure", "Patellofemoral Disorder", "Congenital Disorder",
        "Cartilage Injury", "Ligamentous Injury", "Meniscal Damage"
    ]

    scenario_distribution = {
        "Distal Femoral Fracture": 0.08,
        "Proximal Tibial Fracture": 0.08,
        "Patellar Fracture": 0.08,
        "Osteoarthritis": 0.15,
        "Rheumatoid Arthritis": 0.12,
        "Post-Traumatic Arthritis": 0.12,
        "Primary Bone Tumor": 0.05,
        "Metastatic Lesion": 0.05,
        "Mechanical Failure": 0.07,
        "Patellofemoral Disorder": 0.05,
        "Congenital Disorder": 0.03,
        "Cartilage Injury": 0.07,
        "Ligamentous Injury": 0.07,
        "Meniscal Damage": 0.05
    }

    scenario_distribution = np.array(list(scenario_distribution.values())) / sum(scenario_distribution.values())
    data = []

    for _ in range(num_samples):
        age = np.random.randint(15, 85)
        gender = np.random.choice(["Male", "Female"])
        bmi = round(np.random.uniform(18.5, 40.0), 1)
        activity_level = np.random.choice(["Low", "Moderate", "High"])
        comorbidities = np.random.choice(["None", "Diabetes", "Rheumatoid Arthritis", "Osteoporosis", "Multiple"])
        smoking_status = np.random.choice(["Non-smoker", "Former Smoker", "Current Smoker"])
        alcohol_use = np.random.choice(["No", "Occasional", "Regular"])
        deformity = np.random.choice(["None", "Valgus", "Varus", "Rotational"])
        bone_quality = np.random.choice(["Normal", "Osteoporotic", "Severely Compromised"])

        scenario_idx = np.random.choice(range(len(scenarios)), p=scenario_distribution)
        scenario = scenarios[scenario_idx]

        if scenario in ["Distal Femoral Fracture", "Proximal Tibial Fracture", "Patellar Fracture"]:
            implant = mappings["implants"]["Fractures"][0] if "Distal Femoral" in scenario else \
                mappings["implants"]["Fractures"][1]
        elif scenario in ["Osteoarthritis", "Rheumatoid Arthritis", "Post-Traumatic Arthritis"]:
            implant = mappings["implants"]["Arthritis"][1] if age > 60 or comorbidities in ["Diabetes",
                                                                                            "Rheumatoid Arthritis"] else \
                mappings["implants"]["Arthritis"][0]
        elif scenario in ["Primary Bone Tumor", "Metastatic Lesion"]:
            implant = mappings["implants"]["Tumors"][0]
        elif scenario in ["Mechanical Failure", "Patellofemoral Disorder", "Congenital Disorder"]:
            implant = mappings["implants"]["Instability"][
                0] if bmi > 30 or smoking_status == "Current Smoker" or alcohol_use == "Regular" else \
                mappings["implants"]["Instability"][1]
        elif scenario in ["Cartilage Injury", "Ligamentous Injury", "Meniscal Damage"]:
            implant = mappings["implants"]["Cartilage/Ligament"][0] if activity_level == "High" and (
                    smoking_status != "Current Smoker" and alcohol_use != "Regular") else \
                mappings["implants"]["Cartilage/Ligament"][1]
        else:
            implant = "Total Knee Replacement"

        if implant == "Distal Femoral Replacement":
            procedure = "Bone Grafting" if age > 60 else "ORIF (Open Reduction Internal Fixation)"
        elif implant == "Tibial Plateau Prosthesis":
            procedure = "Bone Grafting" if age > 60 else "ORIF (Open Reduction Internal Fixation)"
        elif implant == "Unicompartmental Knee Replacement":
            procedure = "Partial Knee Arthroplasty" if activity_level == "High" else "General Surgery"
        elif implant == "Total Knee Replacement":
            procedure = "Total Knee Arthroplasty" if age > 70 or comorbidities in ["Diabetes",
                                                                                   "Rheumatoid Arthritis"] else "Partial Knee Arthroplasty"
        elif implant == "Hinged Knee Replacement":
            procedure = "Joint Resurfacing" if deformity in ["Valgus", "Varus"] else "Two-Stage Revision with Spacer"
        elif implant == "Patellofemoral Joint Replacement":
            procedure = "Patellar Resurfacing" if activity_level == "High" and (
                    smoking_status == "Current Smoker" or alcohol_use == "Regular") else "Joint Resurfacing"
        elif implant == "Osteochondral Allograft":
            procedure = "Arthroscopic Meniscal Repair" if bmi > 30 or comorbidities == "Osteoporosis" else "Autologous Chondrocyte Implantation"
        elif implant == "Custom Tumor Prosthesis":
            procedure = "Wide Tumor Excision"
        else:
            procedure = "Wide Tumor Excision"

        data.append({
            "Age": age,
            "Gender": gender,
            "BMI": bmi,
            "Activity Level": activity_level,
            "Comorbidities": comorbidities,
            "Smoking Status": smoking_status,
            "Alcohol Use": alcohol_use,
            "Deformity": deformity,
            "Bone Quality": bone_quality,
            "Scenario": scenario,
            "Recommended Implant": implant,
            "Recommended Procedure": procedure
        })

    return pd.DataFrame(data)


# Generate dataset
synthetic_data = generate_synthetic_data(10000)

# Save dataset
file_path = "../SavedDataset/OptimizedDataSet.csv"
synthetic_data.to_csv(file_path, index=False)


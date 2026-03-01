import pandas as pd
import numpy as np

# Load Monday dataset
file_path = "cicids2017/Monday-WorkingHours.pcap_ISCX.csv"
data = pd.read_csv(file_path)

print("Dataset loaded successfully")
print("Shape:", data.shape)

# Remove leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Check columns
print(data.columns)

# Check labels
print(data['Label'].value_counts())

# Preview data
data.head()

# Replace infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop missing values
data.dropna(inplace=True)

print("After cleaning:", data.shape)

# Keep only numeric features
X = data.select_dtypes(include=[np.number])

print("Numeric shape:", X.shape)

#MODEL TRAINING
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

model.fit(X)

print("Isolation Forest trained successfully")



# Load attack datasets
portscan_path = "cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
ddos_path = "cicids2017/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv"

portscan_data = pd.read_csv(portscan_path)
ddos_data = pd.read_csv(ddos_path)

# Fix column spacing
portscan_data.columns = portscan_data.columns.str.strip()
ddos_data.columns = ddos_data.columns.str.strip()

print("PortScan shape:", portscan_data.shape)
print("DDoS shape:", ddos_data.shape)


# Replace infinite values
portscan_data.replace([np.inf, -np.inf], np.nan, inplace=True)
ddos_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop missing values
portscan_data.dropna(inplace=True)
ddos_data.dropna(inplace=True)

# Select numeric features
X_portscan = portscan_data.select_dtypes(include=[np.number])
X_ddos = ddos_data.select_dtypes(include=[np.number])

print("PortScan numeric:", X_portscan.shape)
print("DDoS numeric:", X_ddos.shape)

# Predict anomalies
portscan_pred = model.predict(X_portscan)
ddos_pred = model.predict(X_ddos)

# Count anomalies
print("PortScan anomalies:", np.sum(portscan_pred == -1))
print("DDoS anomalies:", np.sum(ddos_pred == -1))


# Anomaly scores
portscan_scores = model.decision_function(X_portscan)
ddos_scores = model.decision_function(X_ddos)

print("PortScan score range:", portscan_scores.min(), portscan_scores.max())
print("DDoS score range:", ddos_scores.min(), ddos_scores.max())


def classify_threat(score):
    if score < -0.15:
        return "Zero-Day Attack"
    else:
        return "Insider-Like Anomaly"
portscan_threats = [classify_threat(s) for s in portscan_scores]
ddos_threats = [classify_threat(s) for s in ddos_scores]

print("PortScan threat types:", pd.Series(portscan_threats).value_counts())
print("DDoS threat types:", pd.Series(ddos_threats).value_counts())


#VISUALIZATION
import matplotlib.pyplot as plt

# -------------------------------
# 1. Anomaly Distribution (PortScan)
# -------------------------------
plt.figure()
plt.hist(portscan_scores, bins=50)
plt.title("PortScan Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# 2. Anomaly Distribution (DDoS)
# -------------------------------
plt.figure()
plt.hist(ddos_scores, bins=50)
plt.title("DDoS Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.show()


#SAVING THE MODEL
import joblib
import os

# Create model directory
os.makedirs("models", exist_ok=True)

# Save the trained model
model_path = "models/isolation_forest_model.pkl"
joblib.dump(model, model_path)

print("Model saved successfully at:", model_path)


#TESTING LOADING
# Load model back
loaded_model = joblib.load(model_path)

# Test on few samples
sample_data = X.sample(10)
predictions = loaded_model.predict(sample_data)

print("Sample predictions:", predictions)


# -----------------------------------
# Save detection results
# -----------------------------------

# PortScan results
portscan_results = portscan_data.copy()
portscan_results["anomaly_score"] = portscan_scores
portscan_results["anomaly_label"] = portscan_pred
portscan_results["threat_type"] = portscan_threats

# DDoS results
ddos_results = ddos_data.copy()
ddos_results["anomaly_score"] = ddos_scores
ddos_results["anomaly_label"] = ddos_pred
ddos_results["threat_type"] = ddos_threats

# Create results folder
os.makedirs("results", exist_ok=True)

# Save CSV files
portscan_results.to_csv("results/portscan_detection_results.csv", index=False)
ddos_results.to_csv("results/ddos_detection_results.csv", index=False)

print("Detection results saved successfully")

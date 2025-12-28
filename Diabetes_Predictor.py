# =====================================================
# Diabetes Prediction - End-to-End Data Science Project
# =====================================================

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# =====================================================
# 2. Load Dataset
# =====================================================
data = pd.read_csv("diabetes_prediction_dataset.csv")

print("\nDataset Loaded Successfully")
print(data.head())

# =====================================================
# 3. Data Understanding
# =====================================================
print("\nDataset Shape:", data.shape)
print("\nDataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nStatistical Summary:")
print(data.describe())

# =====================================================
# 4. Data Cleaning
# =====================================================
data.drop_duplicates(inplace=True)
print("\nDuplicates removed. New shape:", data.shape)

# =====================================================
# 5. Exploratory Data Analysis (EDA)
# =====================================================

# Target variable distribution
plt.figure(figsize=(6,4))
sns.countplot(x="diabetes", data=data)
plt.title("Diabetes Distribution")
plt.show()

# Age distribution
plt.figure(figsize=(6,4))
sns.histplot(data["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# BMI vs Diabetes
plt.figure(figsize=(6,4))
sns.boxplot(x="diabetes", y="bmi", data=data)
plt.title("BMI vs Diabetes")
plt.show()

# Correlation heatmap (NUMERIC FEATURES ONLY)
numeric_data = data.select_dtypes(include=[np.number])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# =====================================================
# 6. Encode Categorical Variables
# =====================================================
categorical_cols = ["gender", "smoking_history"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

print("\nCategorical Encoding Completed")

# =====================================================
# 7. Feature Selection
# =====================================================
X = data.drop("diabetes", axis=1)
y = data["diabetes"]

# 🔥 Save feature order (CRITICAL FIX)
feature_order = X.columns.tolist()

# =====================================================
# 8. Train-Test Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# 9. Feature Scaling
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 10. Model Training
# =====================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

model_results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    model_results[name] = acc

    print(f"\n{name}")
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, preds))

# =====================================================
# 11. Select Best Model
# =====================================================
best_model_name = max(model_results, key=model_results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)
print("Best Accuracy:", model_results[best_model_name])

# =====================================================
# 12. Confusion Matrix
# =====================================================
cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =====================================================
# 13. Feature Importance (Random Forest)
# =====================================================
if best_model_name == "Random Forest":
    fi = pd.DataFrame({
        "Feature": feature_order,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(x="Importance", y="Feature", data=fi)
    plt.title("Feature Importance")
    plt.show()

# =====================================================
# 14. Save Model Artifacts
# =====================================================
joblib.dump(best_model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(feature_order, "feature_order.pkl")

print("\nModel, Scaler, Encoders, Feature Order Saved")

# =====================================================
# 15. Sample Prediction (100% ERROR-FREE)
# =====================================================
sample_input = {
    "gender": "Female",
    "age": 45,
    "bmi": 25.3,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "HbA1c_level": 5.8,
    "blood_glucose_level": 120
}

sample_df = pd.DataFrame([sample_input])

# Encode categorical features
for col in categorical_cols:
    sample_df[col] = label_encoders[col].transform(sample_df[col])

# Enforce same column order as training
sample_df = sample_df[feature_order]

# Scale
sample_scaled = scaler.transform(sample_df)

# Predict
prediction = best_model.predict(sample_scaled)

print("\nSample Prediction (0 = No Diabetes, 1 = Diabetes):", prediction[0])

# =====================================================
# END OF PROJECT
# =====================================================

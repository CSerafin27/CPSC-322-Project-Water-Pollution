import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# --------------------------------------------------------
# Load dataset
# --------------------------------------------------------
df = pd.read_csv("/Users/charlesserafin/Desktop/School/2025-2026/CPSC 322/CPSC-322-Project-Water-Pollution/data/water_pollution_disease.csv")

# start with Low for everyone
df["Risk"] = "Low"

# -------- High risk conditions --------
high_mask = (
    (df["Contaminant Level (ppm)"] > 8) |
    (df["Lead Concentration (µg/L)"] > 10) |
    (df["Nitrate Level (mg/L)"] > 50) |
    (df["Bacteria Count (CFU/mL)"] > 500) |
    (df["Turbidity (NTU)"] > 10) |
    (df["Dissolved Oxygen (mg/L)"] < 3) |
    (df["Access to Clean Water (% of Population)"] < 40) |
    (df["Sanitation Coverage (% of Population)"] < 30) |
    (df["Healthcare Access Index (0-100)"] < 30)
)

df.loc[high_mask, "Risk"] = "High"

# -------- Medium risk conditions (only where not already High) --------
medium_mask = (
    (
        (df["Contaminant Level (ppm)"].between(5, 8, inclusive="right")) |
        (df["Lead Concentration (µg/L)"].between(5, 10, inclusive="right")) |
        (df["Nitrate Level (mg/L)"].between(25, 50, inclusive="right")) |
        (df["Bacteria Count (CFU/mL)"].between(100, 500, inclusive="right")) |
        (df["Turbidity (NTU)"].between(5, 10, inclusive="right")) |
        (df["Dissolved Oxygen (mg/L)"].between(3, 5, inclusive="neither")) |
        (df["Access to Clean Water (% of Population)"].between(40, 60, inclusive="left")) |
        (df["Sanitation Coverage (% of Population)"].between(30, 50, inclusive="left")) |
        (df["Healthcare Access Index (0-100)"].between(30, 50, inclusive="left"))
    )
<<<<<<< HEAD
    & ~high_mask  # don’t overwrite High
=======
    & ~high_mask  
>>>>>>> df4f9d7 (Fixed risk level)
)

df.loc[medium_mask, "Risk"] = "Medium"

X = df.select_dtypes(include=["float64", "int64"])
y = df["Risk"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Standardize for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------------
# Evaluation helper
# --------------------------------------------------------
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec_high = precision_score(y_true == "High", y_pred == "High", zero_division=0)
    rec_high = recall_score(y_true == "High", y_pred == "High", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=["Low", "Medium", "High"])
    
    print(f"\n==============================")
    print(f"  {name} RESULTS")
    print(f"==============================")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Precision (High):   {prec_high:.4f}")
    print(f"Recall (High):      {rec_high:.4f}")
    
    # Print confusion matrix neatly
    print("\nConfusion Matrix (rows=true, columns=pred):")
    df_cm = pd.DataFrame(
        cm,
        index=["True Low", "True Medium", "True High"],
        columns=["Pred Low", "Pred Medium", "Pred High"]
    )
    print(df_cm)
    
    return {
        "accuracy": acc,
        "precision_high": prec_high,
        "recall_high": rec_high
    }

results = {}

# --------------------------------------------------------
# Dummy Classifier
# --------------------------------------------------------
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)
results["Dummy Classifier"] = evaluate_model("Dummy Classifier", y_test, y_pred_dummy)

# --------------------------------------------------------
# KNN Classifier
# --------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
results["KNN"] = evaluate_model("KNN (k=5)", y_test, y_pred_knn)

# --------------------------------------------------------
# Decision Tree
# --------------------------------------------------------
tree = DecisionTreeClassifier(criterion="entropy", random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
results["Decision Tree"] = evaluate_model("Decision Tree", y_test, y_pred_tree)

# --------------------------------------------------------
# Random Forest (stand-in for custom RF)
# --------------------------------------------------------
rf = RandomForestClassifier(n_estimators=50, criterion="entropy", random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results["Random Forest"] = evaluate_model("Random Forest", y_test, y_pred_rf)

# --------------------------------------------------------
# Summary Table
# --------------------------------------------------------
summary = pd.DataFrame(results).T
print("\n\n==============================")
print("  SUMMARY TABLE (All Models)")
print("==============================\n")
print(summary.to_string())
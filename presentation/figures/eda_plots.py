import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
# Make sure this path matches your actual filename and location:
df = pd.read_csv("data/water_pollution_disease.csv")
# If you renamed it, use:
# df = pd.read_csv("data/water_pollution_disease.csv")

# 2. Create Risk label using the SAME thresholds as model_comparison.py
df["Risk"] = pd.cut(
    df["Contaminant Level (ppm)"],
    bins=[-1, 5, 8, 10_000],      # Low: <=5, Medium: (5,8], High: >8
    labels=["Low", "Medium", "High"]
)

# Drop any rows where Risk is NaN (edge cases)
df = df.dropna(subset=["Risk"])

print("Risk value counts:")
print(df["Risk"].value_counts())

# 3. Histogram – Contaminant Level
plt.figure(figsize=(8, 5))
sns.histplot(df["Contaminant Level (ppm)"], kde=True, bins=30)
plt.title("Histogram of Contaminant Level (ppm)")
plt.xlabel("Contaminant Level (ppm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Boxplot – Nitrate Level
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Nitrate Level (mg/L)"])
plt.title("Boxplot of Nitrate Level (mg/L)")
plt.xlabel("Nitrate Level (mg/L)")
plt.tight_layout()
plt.show()

# 5. Boxplot – Lead Concentration
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Lead Concentration (µg/L)"])
plt.title("Boxplot of Lead Concentration (µg/L)")
plt.xlabel("Lead Concentration (µg/L)")
plt.tight_layout()
plt.show()

# 6. Risk Level Distribution (now correct & consistent)
plt.figure(figsize=(7, 5))
sns.countplot(x="Risk", data=df, order=["Low", "Medium", "High"])
plt.title("Distribution of Risk Levels")
plt.xlabel("Risk Category")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 7. Correlation Heatmap
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=["float64", "int64"])
corr = numeric_df.corr()

sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap of Water Quality and Health Attributes")
plt.tight_layout()
plt.show()
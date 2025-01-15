import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Sample Data for Tables and Graphs

# Convert to DataFrame
df = pd.read_csv("data/samples.csv")

# Display the Table
print("Data Table:")
print(df)

# Bar Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Category", y="Value", palette="viridis")
plt.title("Bar Plot of Values by Category")
plt.ylabel("Value")
plt.xlabel("Category")
plt.tight_layout()
plt.show()

# Line Plot
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="Category", y="Percentage", marker="o", label="Percentage")
plt.title("Line Plot of Percentage by Category")
plt.ylabel("Percentage")
plt.xlabel("Category")
plt.legend()
plt.tight_layout()
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Value", y="Count", hue="Category", size="Percentage", sizes=(50, 300))
plt.title("Scatter Plot of Value vs Count")
plt.ylabel("Count")
plt.xlabel("Value")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# Add Calculations to Data Table
df["Value_Percentage_Ratio"] = df["Value"] / df["Percentage"]
print("\nUpdated Data Table with Calculations:")
print(df)

# Export Table to CSV (Optional)
print(os.getcwd())
df.to_csv("./src/walkjump/metrics/output_table.csv", index=False)
print("\nTable exported to 'output_table.csv'")

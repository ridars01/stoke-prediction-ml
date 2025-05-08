import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Loading the stroke dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# The first few rows being shown
print("First 5 rows of data:")
print(df.head())

# ID column is not needed so it can be dropped
df = df.drop('id', axis=1)

# Any BMI value that is missing gets filled with a median value across the dataset 
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Labeling all the columns 
label_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store for reference

# Checking the balance of the dataset
print("\nTarget value counts:")
print(df['stroke'].value_counts())

# Plot target balance
sns.countplot(x='stroke', data=df)
plt.title("Class Distribution: Stroke (1) vs No Stroke (0)")
plt.show()

# Features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluation function
def evaluate(model_name, y_true, y_pred):
    print(f"\n=== {model_name} Results ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# Evaluate both models
evaluate("Logistic Regression", y_test, y_pred_lr)
evaluate("Random Forest", y_test, y_pred_rf)

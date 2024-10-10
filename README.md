# CRO-PROJECT
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
df=pd.read_csv("/content/agaricus-lepiota.csv",on_bad_lines='skip')
import pandas as pd
df = pd.read_csv("/content/agaricus-lepiota.csv", on_bad_lines='skip')  # Skips bad lines
df1=pd.read_csv("/content/expanded.csv")
df1 = pd.read_csv("/content/expanded.csv", encoding='ISO-8859-1',on_bad_lines='skip')  # Use the appropriate encoding
df1
df2=pd.read_csv("/content/agaricus.csv",on_bad_lines='skip')
df2
df3=pd.read_csv("/content/agaricus-lepiota.data",sep=",",on_bad_lines='skip')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
df3.describe()
df3.isnull().sum()
column_names = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat"
]

# Load the dataset from the provided CSV file
df4= pd.read_csv("/content/agaricus-lepiota.data",sep=",",on_bad_lines='skip', header=None, names=column_names)
df4.isnull().sum()
# Preprocessing the data
# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(df4.drop("class", axis=1))
y = df4["class"].apply(lambda x: 1 if x == "p" else 0)  # Poisonous = 1, Edible = 0
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)

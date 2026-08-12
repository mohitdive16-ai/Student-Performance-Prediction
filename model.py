import pandas as pd
df = pd.read_csv("student_data.csv")
x = df[["Study_Hours","Attendance","Previous_Marks","Assignments"]]
y = df["Pass"]
print("Features (x):")
print(x)
print()
print("Target (y):")
print(y)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# load Dataset
df = pd.read_csv("student_data.csv")

# Features
X = df[["Study_Hours","Attendance","Previous_Marks","Assignments"]]

# Targets
y = df["Pass"]

# split data
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

# Create Model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# New student
prediction = model.predict([[6, 85, 78, 8]])

# Show result
if prediction[0] == 1:
    print("Student will PASS ✅")
else:
    print("Student will FAIL ❌")


y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test,y_pred)

print("Model Accuracy: ",accuracy)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("student_data.csv")

# creat X and y
X = df[["Study_Hours",
        "Attendance",
        "Previous_Marks",
        "Assignments"]]

y = df["Pass"]

# split the data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# create and Train model

model = LogisticRegression()
model.fit(X_train,y_train)

# check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Model Accuarcy: ",accuracy * 100 , "%") 

# ask the user for information
study_hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter the Attendance: "))
previous_marks = float(input("Enter Previous Marks: "))
assignments = float(input("Ente Assignments: "))

# make prediction

new_student = pd.DataFrame([[
    study_hours,
    attendance,
    previous_marks,
    assignments
]], columns=[
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignments"
])

prediction = model.predict(new_student)

# Display the Result

if prediction[0] == 1:
    print("\n 🎯 Prediction : PASS ✅ ")
else:
    print("\n 🎯 Prediction : FAIL ❌ ")

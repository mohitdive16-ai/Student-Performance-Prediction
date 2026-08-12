import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("student_data.csv")

# PASS VS FAIL
sns.countplot(data = df,x = "Pass")
plt.title("Pass vs Fail Students")
plt.show()

# STUDY HOURS VS PREVIOUS MARKS
plt.scatter(df["Study_Hours"],df["Previous_Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Previous Marks")
plt.title("STUDY HOURS VS PREVIOUS MARKS")
plt.grid(True)
plt.show()

# STUDY HOURS DISTRIBUTION
sns.histplot(data = df , x = "Study_Hours",bins = 5)
plt.title("Study Hours Distibution")
plt.show()

# CORRELATION HEATMAP
sns.heatmap(df.corr(),annot = True)
plt.title("Correlation Heatmap")
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load Digits dataset
digits = load_digits()
X = digits.data
y = digits.target
class_names = digits.target_names

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train Logistic Regression
logreg = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Train Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

#  Generate predictions
y_pred_logreg = logreg.predict(X_test_scaled)
y_pred_knn = knn.predict(X_test_scaled)
y_pred_dt = dt.predict(X_test_scaled)

#  Classification reports
print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_logreg))

print("=== KNN ===")
print(classification_report(y_test, y_pred_knn))

print("=== Decision Tree ===")
print(classification_report(y_test, y_pred_dt))

#  Confusion matrices
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_dt = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
sns.heatmap(cm_logreg, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1,3,2)
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Greens")
plt.title("KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1,3,3)
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Oranges")
plt.title("Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

#Compare accuracies
acc_logreg = accuracy_score(y_test, y_pred_logreg)
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_dt = accuracy_score(y_test, y_pred_dt)

models = ['Logistic Regression', 'KNN', 'Decision Tree']
accuracies = [acc_logreg, acc_knn, acc_dt]

plt.figure(figsize=(8,5))
sns.barplot(x=models, y=accuracies, palette="viridis")
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
for i, acc in enumerate(accuracies):
    plt.text(i, acc+0.02, f"{acc:.2f}", ha='center')
plt.show()

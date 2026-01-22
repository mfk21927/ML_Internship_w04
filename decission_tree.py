import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#  Loading dataset
from sklearn.datasets import load_wine 
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Spliting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train with different max_depth values and tracking accuracy 
from sklearn.tree import DecisionTreeClassifier, plot_tree  
from sklearn.metrics import accuracy_score

depths = [3, 5, 10, None]
accuracy_results = {}

plt.figure(figsize=(8, 5))
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    accuracy_results[str(d)] = acc
    print(f"Max Depth: {d}, Accuracy: {acc:.4f}")

#  Ploting accuracy vs depth 
plt.plot(list(accuracy_results.keys()), list(accuracy_results.values()), marker='o', color='green')
plt.title("Decision Tree: Accuracy vs Max Depth")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

#  Visualize the tree structure for the best depth 
best_depth = 3
final_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final_model.fit(X_train, y_train)

plt.figure(figsize=(20,10))
plot_tree(final_model, feature_names=data.feature_names, class_names=data.target_names, filled=True) # 
plt.title(f"Decision Tree Visualization (max_depth={best_depth})")
plt.savefig("decision_tree_structure.png") 
plt.show()

#  Calculate and visualize feature importance 
importances = final_model.feature_importances_ 
feat_importances = pd.Series(importances, index=data.feature_names)

plt.figure(figsize=(10, 6))
feat_importances.sort_values().plot(kind='barh', color='teal') 
plt.title("Feature Importances in Decision Tree")
plt.xlabel("Importance Score")
plt.show()

# 10. Saving best model 
import pickle
with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
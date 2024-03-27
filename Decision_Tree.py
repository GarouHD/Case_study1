import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Seed for reproducibility
RANDOM_STATE = 40

# Load data
df = pd.read_csv('Data/train.csv').to_numpy()
X = df[:, :-1]
Y = df[:, -1]

# Preprocess data, X Maybe

# Decision Tree Model
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)

# k-fold cross-validation
k = 10
scores = cross_val_score(dt, X, Y, cv=k, scoring='accuracy')

print(f'Accuracy for each fold: {scores}')
print(f'Mean Accuracy: {scores.mean() * 100:.2f}%')





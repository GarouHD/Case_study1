import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Seed for reproducibility
RANDOM_STATE = 40

# Load data
df = pd.read_csv('Data/train.csv').to_numpy()
X = df[:, :-1]
Y = df[:, -1]

# Preprocess data
X = preprocessing.StandardScaler().fit_transform(X)

# Logistic Regression Model
lrm = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=5, penalty='l2')

# k-fold cross-validation
k = 10
scores = cross_val_score(lrm, X, Y, cv=k, scoring='accuracy')

print(f'Accuracy for each fold: {scores}')
print(f'Mean Accuracy: {scores.mean() * 100:.2f}%')





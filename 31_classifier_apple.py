"""
Case study 1,
Ziyad Sbeih, zxs473
Stephanie little, sal164
"""
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Seed for reproducibility
RANDOM_STATE = 40

# Load training data
df_train = pd.read_csv('Data/train.csv').to_numpy()
X_train = df_train[:, :-1]
Y_train = df_train[:, -1]

# Load Testing data
df_test = pd.read_csv('Data/test.csv').to_numpy()
X_test = df_test[:, :-1]
Y_test = df_test[:, -1]

# Logistic Regression Model with hyper tuned params
svm = SVC(random_state=RANDOM_STATE, C=1000, gamma=0.01, kernel='rbf')

# Fit model to training data
svm.fit(X_train, Y_train)

# Get predictions
Y_pred = svm.predict(X_train)

# Get accuracy scores
final_acc = accuracy_score(Y_test, Y_pred)

print(f'Test Accuracy: {final_acc*100:.2f}%')
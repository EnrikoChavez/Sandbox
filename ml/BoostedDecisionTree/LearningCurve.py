from ucimlrepo import fetch_ucirepo
import sklearn
from pprint import pprint
from sklearn.model_selection import KFold
import numpy as np
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.ensemble import GradientBoostingClassifier

# fetch dataset
# yeast = fetch_ucirepo(id=110)
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

X = breast_cancer_wisconsin_diagnostic.data.features.to_numpy()
y = breast_cancer_wisconsin_diagnostic.data.targets.to_numpy()

X_train = X
y_train = y

#Boosted DecisionTreeClassifier
#TODO look up boosted decision trees
estimator = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

#learning curve
train_sizes = [1, 2, 5, 10, 15, 25, 50, 75, 100, 150, 200, 250, 300, 430]
le = LabelEncoder()
le.fit(y.flatten())
y_encoded = le.transform(y.flatten())
train_sizes, train_scores, validation_scores = learning_curve(estimator = estimator, X = X, y = y_encoded, train_sizes = train_sizes, cv = 5, scoring = 'neg_mean_squared_error')
train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a "" model', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,0.25)
plt.show()

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
estimator = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

#validation curve
disp = ValidationCurveDisplay.from_estimator(estimator = estimator, X=X, y=y.flatten(), param_name="n_estimators", param_range=np.logspace(0, 3, 4, dtype='int'))
disp.ax_.set_title("Validation Curve for Boosted Trees")
disp.ax_.set_xlabel(r"number of estimators")
disp.ax_.set_ylim(0.0, 1.1)
plt.show()
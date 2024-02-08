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

# fetch dataset
# yeast = fetch_ucirepo(id=110)
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

X = breast_cancer_wisconsin_diagnostic.data.features.to_numpy()
y = breast_cancer_wisconsin_diagnostic.data.targets.to_numpy()

X_train = X
y_train = y

#KNearest Neighbor
k = 10 #different Ks for analysis
estimator = KNeighborsClassifier(n_neighbors=k)
#TODO algorithm to use need to do validation curve independantly

#validation curve
disp = ValidationCurveDisplay.from_estimator(estimator = estimator, X=X, y=y.flatten(), param_name="n_neighbrs", param_range=[1,2,4,10,50,100])
disp.ax_.set_title("Validation Curve for SVM with an RBF kernel")
disp.ax_.set_xlabel(r"gamma (inverse radius of the RBF kernel)")
disp.ax_.set_ylim(0.0, 1.1)
plt.show()
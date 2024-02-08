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

#DecisionTreeClassifier
clf = tree.DecisionTreeClassifier()
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
clf = tree.DecisionTreeClassifier(ccp_alpha=ccp_alphas[int(len(ccp_alphas)/2)]) # getting the median alpha (not too little or not too much pruning)
# clf.fit(X,y)
# tree.plot_tree(clf)
# plt.show()

#Boosted DecisionTreeClassifier
#TODO look up boosted decision trees

#Neural Network Classifier
nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=800) #for analysis different layers, diff algs

#Support Vector Machine
svc_clf = svm.SVC() #TODO choose two different kernel functions (for analysis)

#KNearest Neighbor
k = 10 #different Ks for analysis
knn_clf = KNeighborsClassifier(n_neighbors=k)

#learning curve
# train_sizes = [1, 2, 5, 10, 15, 25, 50, 75, 100, 150, 200, 250, 300, 430]
# estimator = nn_clf
# le = LabelEncoder()
# le.fit(y.flatten())
# y_encoded = le.transform(y.flatten())
# train_sizes, train_scores, validation_scores = learning_curve(estimator = estimator, X = X, y = y_encoded, train_sizes = train_sizes, cv = 5, scoring = 'neg_mean_squared_error')
# train_scores_mean = -train_scores.mean(axis = 1)
# validation_scores_mean = -validation_scores.mean(axis = 1)

# plt.plot(train_sizes, train_scores_mean, label = 'Training error')
# plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
# plt.ylabel('MSE', fontsize = 14)
# plt.xlabel('Training set size', fontsize = 14)
# plt.title('Learning curves for a "" model', fontsize = 18, y = 1.03)
# plt.legend()
# plt.ylim(0,1)
# plt.show()

#validation curve
# train_scores, validation_scores = validation_curve(svm.SVC(kernel="linear"), X, y.flatten(), param_name="C", param_range=np.logspace(-7, 3, 3))
# disp = ValidationCurveDisplay.from_estimator(estimator = nn_clf, X=X, y=y.flatten(), param_name="alpha", param_range=np.logspace(-7, 3, 10))
# disp.ax_.set_title("Validation Curve for SVM with an RBF kernel")
# disp.ax_.set_xlabel(r"gamma (inverse radius of the RBF kernel)")
# disp.ax_.set_ylim(0.0, 1.1)
# plt.show()
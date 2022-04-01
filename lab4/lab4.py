from sklearn import datasets
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle

data_bc = datasets.load_breast_cancer()

X_bc = data_bc.data[:, (3, 4)]
y_bc = data_bc.target

X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(X_bc, y_bc, test_size=0.2, random_state=42)

bc_svm_clf_without_scaler = Pipeline([("linear_svc", LinearSVC(C=1,loss="hinge",random_state=42)),])
bc_svm_clf_with_scaler = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge",
                                                                                          random_state=42)), ])
bc_svm_clf_without_scaler.fit(X_bc_train, y_bc_train)
bc_svm_clf_with_scaler.fit(X_bc_train, y_bc_train)

bc_acc_svm_unscaled_train = bc_svm_clf_without_scaler.score(X_bc_train, y_bc_train)
bc_acc_svm_unscaled_test = bc_svm_clf_without_scaler.score(X_bc_test, y_bc_test)
bc_acc_svm_scaled_train = bc_svm_clf_with_scaler.score(X_bc_train, y_bc_train)
bc_acc_svm_scaled_test = bc_svm_clf_with_scaler.score(X_bc_test, y_bc_test)

bc_acc = [bc_acc_svm_unscaled_train, bc_acc_svm_unscaled_test, bc_acc_svm_scaled_train, bc_acc_svm_scaled_test]

print(bc_acc)

with open('bc_acc.pkl', 'wb') as f:
    pickle.dump(bc_acc, f)

data_iris = datasets.load_iris()

X_ir = data_iris.data[:, (2, 3)]
y_ir = (data_iris["target"] == 2).astype(np.int8)

X_ir_train, X_ir_test, y_ir_train, y_ir_test = train_test_split(X_ir, y_ir, test_size=0.2)

print(data_iris.feature_names)

iris_svm_clf_without_scaler = Pipeline([("linear_svc", LinearSVC(C=1,loss="hinge",random_state=42)),])
iris_svm_clf_with_scaler = Pipeline([("scaler", StandardScaler()),("linear_svc", LinearSVC(C=1,loss="hinge",
                                                                                           random_state=42)), ])
iris_svm_clf_without_scaler.fit(X_ir_train,y_ir_train)
iris_svm_clf_with_scaler.fit(X_ir_train,y_ir_train)

iris_acc_svm_unscaled_train = iris_svm_clf_without_scaler.score(X_ir_train, y_ir_train)
iris_acc_svm_unscaled_test = iris_svm_clf_without_scaler.score(X_ir_test, y_ir_test)
iris_acc_svm_scaled_train = iris_svm_clf_with_scaler.score(X_ir_train, y_ir_train)
iris_acc_svm_scaled_test = iris_svm_clf_with_scaler.score(X_ir_test, y_ir_test)

iris_acc=[iris_acc_svm_unscaled_train, iris_acc_svm_unscaled_test, iris_acc_svm_scaled_train, iris_acc_svm_scaled_test]

print(iris_acc)

with open('iris_acc.pkl', 'wb') as f:
    pickle.dump(iris_acc, f)
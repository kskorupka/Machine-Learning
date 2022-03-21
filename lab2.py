from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import SGDClassifier 
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import pickle

mnist = fetch_openml('mnist_784', version=1)
X,y = mnist["data"], mnist["target"]
print((np.array(X.loc[42]).reshape(28,28)>0).astype(int))

y_sorted = y.sort_values()
X_sorted = X.reindex(y_sorted.index)
print(X_sorted)
print(y_sorted)

X_train_sorted, X_test_sorted = X_sorted[:56000], X_sorted[56000:]
y_train_sorted, y_test_sorted = y_sorted[:56000], y_sorted[56000:]
print(X_train_sorted.shape, y_train_sorted.shape)
print(X_test_sorted.shape, y_test_sorted.shape)
print('y_train: ', np.unique(y_train_sorted))
print('y_test: ', np.unique(y_test_sorted))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print('y_train: ', np.unique(y_train))
print('y_test: ', np.unique(y_test))

y_train_0 = (y_train == '0')
y_test_0 = (y_test == '0')
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)
accuracy_train = sgd_clf.score(X_train, y_train_0)
sgd_clf.fit(X_test, y_test_0)
accuracy_test = sgd_clf.score(X_test, y_test_0)
print(accuracy_train)
print(accuracy_test)
accuracy = [accuracy_train, accuracy_test]
print(accuracy)
with open('sgd_acc.pkl', 'wb') as f:
    pickle.dump(accuracy, f)

sgd_clf.fit(X_train, y_train_0)
cvs = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs=-1)
print(cvs)
with open('sgd_cva.pkl', 'wb') as f:
    pickle.dump(cvs, f)

sgd_m_clf = SGDClassifier(random_state=42,n_jobs=-1)
sgd_m_clf.fit(X_train, y_train)
y_train_predict = cross_val_predict(sgd_m_clf, X_train, y_train, cv=3, n_jobs=-1)
conf_mx = confusion_matrix(y_train, y_train_predict)
print(conf_mx)
with open('sgd_cmx.pkl', 'wb') as f:
    pickle.dump(conf_mx, f)






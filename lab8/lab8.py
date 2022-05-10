from sklearn import datasets
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

data_breast_cancer = datasets.load_breast_cancer()
data_iris = load_iris()
pca_db = PCA(n_components=0.9)
pca_di = PCA(n_components=0.9)

X_reduced_db = pca_db.fit_transform(data_breast_cancer.data)

X_reduced_di = pca_di.fit_transform(data_iris.data)

print(X_reduced_db.shape)
print(data_breast_cancer.data.shape)
print(X_reduced_di.shape)
print(data_iris.data.shape)

sca = StandardScaler()

sca.fit(data_breast_cancer.data)

X_scaled_db = sca.transform(data_breast_cancer.data)

sca.fit(data_iris.data)

pca_db_scaled = PCA(n_components=0.9)

pca_di_scaled = PCA(n_components=0.9)

X_scaled_di = sca.transform(data_iris.data)

X_reduced_db_scaled = pca_db_scaled.fit_transform(X_scaled_db)

X_reduced_di_scared = pca_di_scaled.fit_transform(X_scaled_di)

print(X_reduced_db_scaled.shape)

print(X_reduced_di_scared.shape)

with open('pca_bc.pkl','wb') as f:
    pickle.dump(pca_db_scaled.explained_variance_ratio_,f)

with open('pca_ir.pkl','wb') as f:
    pickle.dump(pca_di_scaled.explained_variance_ratio_,f)

with open('pca_bc.pkl','rb') as f:
    bc_test_list = pickle.load(f)

with open('pca_ir.pkl','rb') as f:
    ir_test_list = pickle.load(f)

print(bc_test_list)
print(ir_test_list)
print(pca_di_scaled.components_)

di_idx = []
db_idx = []

for row in pca_di_scaled.components_:
    di_idx.append(np.argmax(row))
    print(np.max(row), np.argmax(row))

for row in pca_db_scaled.components_:
    db_idx.append(np.argmax(row))
    print(np.max(row), np.argmax(row))

print(di_idx)
print(db_idx)

with open('idx_bc.pkl', 'wb') as f:
    pickle.dump(db_idx, f)

with open('idx_ir.pkl', 'wb') as f:
    pickle.dump(di_idx, f)

with open('idx_bc.pkl', 'rb') as f:
    idx_bc_test = pickle.load(f)

with open('idx_ir.pkl', 'rb') as f:
    idx_ir_test = pickle.load(f)

print(idx_bc_test)
print(idx_ir_test)

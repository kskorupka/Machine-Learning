import graphviz
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from sklearn.metrics import f1_score, mean_squared_error
import pickle
import matplotlib.pyplot as plt

#Prepare data
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)

size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})

#df.plot.scatter(x='x',y='y')

#Classification
#Split data
X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(data_breast_cancer.data, data_breast_cancer.target, test_size=0.2, random_state=42)

f1_list = []
best_depth_bc = 1

tree_clf_bc = DecisionTreeClassifier(max_depth=1, random_state=42)

tree_clf_bc.fit(X_bc_train, y_bc_train)

y_bc_pred_train = tree_clf_bc.predict(X_bc_train)

y_bc_pred_test = tree_clf_bc.predict(X_bc_test)

f1_list.append((f1_score(y_bc_train, y_bc_pred_train), f1_score(y_bc_test, y_bc_pred_test)))

#Find the best depth
for depth in range(2,20):
    tree_clf_bc = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_clf_bc.fit(X_bc_train, y_bc_train)
    y_bc_pred_train = tree_clf_bc.predict(X_bc_train)
    y_bc_pred_test = tree_clf_bc.predict(X_bc_test)
    f1_bc_score_train = f1_score(y_bc_train, y_bc_pred_train)
    f1_bc_score_test = f1_score(y_bc_test, y_bc_pred_test)
    if f1_bc_score_test < f1_list[depth-2][1]:
        #add data to check if it's rational
        f1_list.append((f1_bc_score_train, f1_bc_score_test))
        break
    elif (f1_bc_score_train,f1_bc_score_test) > f1_list[depth-2]:
        best_depth_bc = depth
    f1_list.append((f1_bc_score_train, f1_bc_score_test))

for row in f1_list:
    print(row)

print(best_depth_bc)

tree_clf_bc = DecisionTreeClassifier(max_depth=best_depth_bc, random_state=42)
tree_clf_bc.fit(X_bc_train, y_bc_pred_train)

#Export tree_clf graph to .png
f = "bc"

export_graphviz(tree_clf_bc, out_file=f, feature_names=data_breast_cancer.feature_names,
                class_names=data_breast_cancer.target_names, rounded=True, filled=True)
#print(f)

print(graphviz.render('dot','png', f))

graph = graphviz.Source.from_file(f)

y_bc_pred_train = tree_clf_bc.predict(X_bc_train)
y_bc_pred_test = tree_clf_bc.predict(X_bc_test)

f1_bc_score_train = f1_score(y_bc_train, y_bc_pred_train)
f1_bc_score_test = f1_score(y_bc_test, y_bc_pred_test)

acc_bc_train = tree_clf_bc.score(X_bc_train, y_bc_train)
acc_bc_test = tree_clf_bc.score(X_bc_test, y_bc_test)

print('Accuracy: ', acc_bc_train, acc_bc_test)

f1acc_tree = [best_depth_bc, f1_bc_score_train, f1_bc_score_test, acc_bc_train, acc_bc_test]

print('f1acc_tree: ', f1acc_tree)

with open('f1acc_tree.pkl', 'wb') as f:
    pickle.dump(f1acc_tree, f)

with open('f1acc_tree.pkl', 'rb') as f:
    f1acc_test = pickle.load(f)

print('f1acc_tree from pickle: ', f1acc_test)

#Regression
tree_reg_df = DecisionTreeRegressor(max_depth=1, random_state=42)

#Split data
X_df_train, X_df_test, y_df_train, y_df_test = train_test_split(df[['x']], df[['y']], test_size=0.2, random_state=42)

tree_reg_df.fit(X_df_train, y_df_train)

y_df_pred_train = tree_reg_df.predict(X_df_train)
y_df_pred_test = tree_reg_df.predict(X_df_test)

mse_df_train = mean_squared_error(y_df_train, y_df_pred_train)
mse_df_test = mean_squared_error(y_df_test, y_df_pred_test)

#print(mse_df_train, mse_df_test)

mse_list = []

best_depth_reg = 1
mse_list.append((mse_df_train, mse_df_test))

#Find the best depth
for depth in range(2,20):
    tree_reg_df = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree_reg_df.fit(X_df_train, y_df_train)
    y_df_pred_train = tree_reg_df.predict(X_df_train)
    y_df_pred_test = tree_reg_df.predict(X_df_test)
    mse_df_train = mean_squared_error(y_df_train, y_df_pred_train)
    mse_df_test = mean_squared_error(y_df_test, y_df_pred_test)
    if mse_df_test > mse_list[depth-2][1]:
        #add data to check if it's rational
        mse_list.append((mse_df_train, mse_df_test))
        break
    elif (mse_df_train, mse_df_test) < mse_list[depth-2]:
        best_depth_reg = depth
    mse_list.append((mse_df_train, mse_df_test))

for row in mse_list:
    print(row)

print(best_depth_reg)

tree_reg_df = DecisionTreeRegressor(max_depth=best_depth_reg, random_state=42)

tree_reg_df.fit(X_df_train, y_df_train)

y_df_pred_train = tree_reg_df.predict(X_df_train)
y_df_pred_test = tree_reg_df.predict(X_df_test)

mse_df_train = mean_squared_error(y_df_train, y_df_pred_train)
mse_df_test = mean_squared_error(y_df_test, y_df_pred_test)
#plt.scatter(df.x, df.y)
#plt.scatter(X_df_train, y_df_pred_train, color='r')
#plt.scatter(X_df_test, y_df_pred_test, color='r')

#Export tree_reg graph to .png
f_df = "reg"

export_graphviz(tree_reg_df, out_file=f_df, rounded=True, filled=True)
#print(f_df)

print(graphviz.render('dot','png', f_df))

mse_tree = [best_depth_reg, mse_df_train, mse_df_test]

print('mse_tree: ', mse_tree)

with open('mse_tree.pkl', 'wb') as f:
    pickle.dump(mse_tree, f)

with open('mse_tree.pkl', 'rb') as f:
    mse_tree_test = pickle.load(f)
print('mse_tree from pickle: ', mse_tree_test)
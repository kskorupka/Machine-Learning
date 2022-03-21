import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sklearn.neighbors
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import pickle

size = 300
X = np.random.rand(size)*5-2.5
w4,w3,w2,w1,w0 = 1,2,1,-4,2
y=w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.to_csv('dane_do_regresji.csv', index=None)
df.plot.scatter(x='x', y='y')

#Split sets
X_train, X_test, y_train, y_test = train_test_split(df[['x']], df[['y']], test_size=0.2, random_state=42)

#Perform Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

#train
y_train_pred = lin_reg.predict(X_train)
plt.scatter(df.x, df.y, color ='b')
plt.plot(X_train, y_train_pred, color ='r')
lin_reg_train_mse = mean_squared_error(y_train, y_train_pred)
print(lin_reg_train_mse)

#test
y_test_pred = lin_reg.predict(X_test)
plt.scatter(df.x, df.y, color ='b')
plt.plot(X_test, y_test_pred, color ='r')
lin_reg_test_mse = mean_squared_error(y_test, y_test_pred)
print(lin_reg_test_mse)

#Perform KNN Regression
knn_3_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn_5_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
knn_3_reg.fit(X_train,y_train)
knn_5_reg.fit(X_train,y_train)

#train
y_knn3_train_pred = knn_3_reg.predict(X_train)
y_knn5_train_pred = knn_5_reg.predict(X_train)

plt.scatter(df.x, df.y, color ='b')
plt.scatter(X_train, y_knn3_train_pred, color ='r')
knn_3_train_mse = mean_squared_error(y_train, y_knn3_train_pred)
print(knn_3_train_mse)

plt.scatter(df.x, df.y, color ='b')
plt.scatter(X_train, y_knn5_train_pred, color ='r')
knn_5_train_mse = mean_squared_error(y_train, y_knn5_train_pred)
print(knn_5_train_mse)

#test
y_knn3_test_pred = knn_3_reg.predict(X_test)
y_knn5_test_pred = knn_5_reg.predict(X_test)

plt.scatter(df.x, df.y, color ='b')
plt.scatter(X_test, y_knn3_test_pred, color ='r')
knn_3_test_mse = mean_squared_error(y_test, y_knn3_test_pred)
print(knn_3_test_mse)

plt.scatter(df.x, df.y, color ='b')
plt.scatter(X_test, y_knn5_test_pred, color ='r')
knn_5_test_mse = mean_squared_error(y_test, y_knn5_test_pred)
print(knn_5_test_mse)

#Perform Polynomial Regression
poly_feature_2 = PolynomialFeatures(degree=2, include_bias=False)
poly_feature_3 = PolynomialFeatures(degree=3, include_bias=False)
poly_feature_4 = PolynomialFeatures(degree=4, include_bias=False)
poly_feature_5 = PolynomialFeatures(degree=5, include_bias=False)

X_train_poly_2 = poly_feature_2.fit_transform(X_train)
X_train_poly_3 = poly_feature_3.fit_transform(X_train)
X_train_poly_4 = poly_feature_4.fit_transform(X_train)
X_train_poly_5 = poly_feature_5.fit_transform(X_train)

poly_2_reg = LinearRegression()
poly_3_reg = LinearRegression()
poly_4_reg = LinearRegression()
poly_5_reg = LinearRegression()

poly_2_reg.fit(X_train_poly_2, y_train)
poly_3_reg.fit(X_train_poly_3, y_train)
poly_4_reg.fit(X_train_poly_4, y_train)
poly_5_reg.fit(X_train_poly_5, y_train)

y_train_pred_poly2 = poly_2_reg.predict(X_train_poly_2)
y_train_pred_poly3 = poly_3_reg.predict(X_train_poly_3)
y_train_pred_poly4 = poly_4_reg.predict(X_train_poly_4)
y_train_pred_poly5 = poly_5_reg.predict(X_train_poly_5)

plt.scatter(df.x, df.y, color ='b')
plt.scatter(X_train, y_train_pred_poly2, color ='r')
poly2_train_mse = mean_squared_error(y_train, y_train_pred_poly2)
print(poly2_train_mse)

plt.scatter(df.x, df.y, color ='b')
plt.scatter(X_train, y_train_pred_poly3, color ='r')
poly3_train_mse = mean_squared_error(y_train, y_train_pred_poly3)
print(poly3_train_mse)

plt.scatter(df.x, df.y, color ='b')
plt.scatter(X_train, y_train_pred_poly4, color ='r')
poly4_train_mse = mean_squared_error(y_train, y_train_pred_poly4)
print(poly4_train_mse)

plt.scatter(df.x, df.y, color ='b')
plt.scatter(X_train, y_train_pred_poly5, color ='r')
poly5_train_mse = mean_squared_error(y_train, y_train_pred_poly5)
print(poly5_train_mse)

#test
X_test_poly_2 = poly_feature_2.fit_transform(X_test)
X_test_poly_3 = poly_feature_3.fit_transform(X_test)
X_test_poly_4 = poly_feature_4.fit_transform(X_test)
X_test_poly_5 = poly_feature_5.fit_transform(X_test)

y_test_pred_poly2 = poly_2_reg.predict(X_test_poly_2)
y_test_pred_poly3 = poly_3_reg.predict(X_test_poly_3)
y_test_pred_poly4 = poly_4_reg.predict(X_test_poly_4)
y_test_pred_poly5 = poly_5_reg.predict(X_test_poly_5)

plt.scatter(df.x, df.y, color ='b')
plt.scatter(X_test, y_test_pred_poly2, color ='r')
poly2_test_mse = mean_squared_error(y_test, y_test_pred_poly2)
print(poly2_test_mse)

plt.scatter(df.x, df.y, color ='b')
plt.scatter(X_test, y_test_pred_poly3, color ='r')
poly3_test_mse = mean_squared_error(y_test, y_test_pred_poly3)
print(poly3_test_mse)

plt.scatter(df.x, df.y, color ='b')
plt.scatter(X_test, y_test_pred_poly4, color ='r')
poly4_test_mse = mean_squared_error(y_test, y_test_pred_poly4)
print(poly4_test_mse)

plt.scatter(df.x, df.y, color ='b')
plt.scatter(X_test, y_test_pred_poly5, color ='r')
poly5_test_mse = mean_squared_error(y_test, y_test_pred_poly5)
print(poly5_test_mse)

data = {'train_mse':[lin_reg_train_mse,knn_3_train_mse, knn_5_train_mse, poly2_train_mse, poly3_train_mse, poly4_train_mse, poly5_train_mse],
        'test_mse':[lin_reg_test_mse, knn_3_test_mse, knn_5_test_mse, poly2_test_mse, poly3_test_mse, poly4_test_mse, poly5_test_mse]}
reg_df = pd.DataFrame(data)
reg_df.index=['lin_reg','knn_3_reg', 'knn_5_reg', 'poly_2_reg', 'poly_3_reg', 'poly_4_reg', 'poly_5_reg']
print(reg_df)

with open('mse.pkl', 'wb') as f:
    pickle.dump(reg_df, f)

reg = [(lin_reg, None), (knn_3_reg, None), (knn_5_reg, None), (poly_2_reg, poly_feature_2),(poly_3_reg, poly_feature_3), (poly_4_reg, poly_feature_4),
(poly_5_reg, poly_feature_5)]

with open('reg.pkl', 'wb') as f:
    pickle.dump(reg_df, f)





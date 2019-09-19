import pandas as pd
df = pd.read_excel (r'vaemo.xlsx')
import random
df.sample(frac=1)

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

start=1106
end=1356
y=df.iloc[:,1:2]
X=df.iloc[:,3:4]

X_train, X_test, y_train, y_test     = train_test_split(X, y, test_size=0.2, random_state=1)

#X_train, X_val, y_train, y_val     = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
'''
from sklearn.ensemble import RandomForestRegressor
depthList={1,2,4,5,8,10}
est={5,10,20,40,80,100,200}
variance_score=0
bestregr=RandomForestRegressor(max_depth=1, random_state=0,
                             n_estimators=5)
for i in depthList:
    for j in est:
        print i,j

        regr = RandomForestRegressor(max_depth=i, random_state=0,
                             n_estimators=j)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_val)
        print r2_score(y_val,y_pred)
        if(r2_score(y_val,y_pred)>variance_score):
            variance_score=r2_score(y_val,y_pred)
            bestregr=regr

# training a DescisionTreeClassifier

print bestregr.max_depth, bestregr.n_estimators
y_pred = bestregr.predict(X_test)
'''
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
poly = PolynomialFeatures(degree=1)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)
y_train = poly.fit_transform(y_train)
y_test = poly.fit_transform(y_test)

print(X_train)

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
regr = linear_model.LinearRegression()
#regr = Lasso(alpha=0.01)
# Train the model using the training sets

#regr = RandomForestRegressor(max_depth=4, random_state=1,
#                             n_estimators=10)
#regr.fit(X_train, y_train)
regr.fit(X_train, y_train)

# Make predictions using the testing set

# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean squared error
y_pred = regr.predict(X_test)

#print regr.score(y_test,y_pred)
print("TEST Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('TEST Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')
#plt.show()#print y_test.shape
# Plot outputs
#plt.plot(X_test[:0], X_test[:1], color='blue', linewidth=1)





'''
'''

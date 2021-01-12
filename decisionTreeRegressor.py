import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from yellowbrick.model_selection import FeatureImportances

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('audi.csv')
# pd.set_option('display.max_columns',9)
#REPLACING STRINGS INTO ZEROS AND ON ES
df['transmission'].replace(['Manual','Automatic','Semi-Auto'],['0','1','2'], inplace=True)
df['fuelType'].replace(['Petrol','Diesel','Hybrid'],['0','1','2'], inplace=True)
df['model'].replace([' A1',' A6',' A4',' A3',' Q3',' Q5',' A5',' S4',' Q2',' A7',' TT',' Q7',' RS6',' RS3',' A8',' Q8',' RS4',' RS5',' R8',' SQ5',' S8',' SQ7',' S3',' S5',' A2',' RS7'],['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'], inplace=True)

print(df["engineSize"].max())

# print(df.info())
X = df[['model','year','transmission','engineSize']]
Y = df[['price']]
# X = df.iloc[:,[0,7]].values
# y = df.iloc[:,[8]].values
# X = X.iloc[:,[0,1,2,3,4,5,6,7]].values
# y = y.iloc[:,[0]].values
scaler = preprocessing.StandardScaler().fit(X)
data_scaled = scaler.transform(X)
X_scaled = data_scaled

scaler = preprocessing.StandardScaler().fit(Y)
data_scaled = scaler.transform(Y)
y_scaled = data_scaled


# print(X_scaled)
X_full_train, X_full_test, Y_full_train, Y_full_test = train_test_split(X_scaled, y_scaled, test_size = 0.33, random_state = 0)
print("X Train",len(X_full_train))
print("X Test",len(X_full_test))

model = DecisionTreeRegressor()
model.fit(X_full_train,Y_full_train)
y_predict = model.predict(X_full_test)

print('r2 score after 1 trial=',r2_score(Y_full_test,y_predict))
print('MAE score after 1 trial=',mean_absolute_error(Y_full_test,y_predict))
print('RMSE score after 1 trial',np.sqrt(mean_squared_error(Y_full_test,y_predict)))

# iteration = []
# for i in range (len(y_predict)):
#     iteration.append(i)
#
# plt.plot(iteration[0:100], y_predict[0:100], label="Predicted")
# plt.plot(iteration[0:100], Y_full_test[0:100].flatten(), label="Actual")
# plt.legend()
# plt.title("Actual vs Prediction for Decision Tree Regression")



meanr2 = 0
meanMAE = 0
meanRMSE = 0
for i in range(100):
    model.fit(X_full_train,Y_full_train)
    y_predict = model.predict(X_full_test)
    meanr2 += r2_score(Y_full_test,y_predict)
    meanMAE += mean_absolute_error(Y_full_test,y_predict)
    meanRMSE += np.sqrt(mean_squared_error(Y_full_test,y_predict))


print('mean r2 score after 100 trial=',meanr2/100)
print('mean MAE score after 100 trial=',meanMAE/100)
print('mean RMSE score after 100 trial',meanRMSE/100)
# print(model.feature_importances_)
# viz = FeatureImportances(model)
# viz.fit(X_full_test,Y_full_test)
# viz.show()

np.random.seed(19680801)


plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
people = ('Mpg', 'Year', 'Model', 'Engine Size', 'Mileage', 'Transmission', 'Tax', 'Fuel Type')
y_pos = np.arange(len(people))
performance = [100,58,30,28,8,7,5,2]
color = ['#77AC30','#0072BD','#4DBEEE','#EDB120','#7E2F8E','#A2142F','#77AC30','#0072BD']

ax.barh(y_pos, performance, align='center', color=color)
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Relative Importances')
ax.set_title('Feature Importances of 8 Features using DecisionTreeRegressor')

plt.show()

#
# importance = model.feature_importances_
#
# for i,v in enumerate(importance):
# 	print('Feature: %0d, Score: %.5f' % (i,v))

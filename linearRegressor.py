import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report
# from yellowbrick.model_selection import FeatureImportances

from sklearn.linear_model import LinearRegression

import seaborn as sns
import warnings

import numpy as np

from scipy.stats import ttest_ind
from scipy import stats
warnings.filterwarnings("ignore")
df = pd.read_csv('audi.csv')
# pd.set_option('display.max_columns',9)
#REPLACING STRINGS INTO ZEROS AND ON ES

df['transmission'].replace(['Manual','Automatic','Semi-Auto'],['0','1','2'], inplace=True)
df['fuelType'].replace(['Petrol','Diesel','Hybrid'],['0','1','2'], inplace=True)
df['model'].replace(
    [' A1',' A6',' A4',' A3',' Q3',' Q5',' A5',' S4',' Q2',' A7',' TT',' Q7',' RS6',' RS3',' A8',' Q8',' RS4',' RS5',' R8',' SQ5',' S8',' SQ7',' S3',' S5',' A2',' RS7'],
    ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'], inplace=True)
# 3456
# print(df.info())
X = df[['model','year','transmission','mileage','fuelType','tax','mpg','engineSize']]
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


X_full_train, X_full_test, Y_full_train, Y_full_test = train_test_split(X_scaled, y_scaled, test_size = 0.33, random_state = 0)



print("X Train",len(X_full_train))
print("X Test",len(X_full_test))



reg = LinearRegression()

reg.fit(X_full_train,Y_full_train)
score = reg.score(X_full_test, Y_full_test)
y_predict = reg.predict(X_full_test)

r2= r2_score(Y_full_test,y_predict)
mae=mean_absolute_error(Y_full_test,y_predict)
rmse=np.sqrt(mean_squared_error(Y_full_test,y_predict))

iteration = []
for i in range (len(y_predict)):
    iteration.append(i)

plt.plot(iteration[0:100], y_predict[0:100], label="Predicted")
plt.plot(iteration[0:100], Y_full_test[0:100].flatten(), label="Actual")
plt.legend()
plt.title("Actual vs Prediction for Linear Regression")
plt.show()


print('R2 score from 1 trial ', r2)
print('Absolute Error from 1 trial ', mae)
print('Squarred Error from 1 trial ', rmse)


meanr2 = 0
meanscore = 0
meanMAE = 0
meanRMSE = 0
for i in range(100):
    reg.fit(X_full_train,Y_full_train)
    y_predict = reg.predict(X_full_test)
    r2= r2_score(y_predict,Y_full_test)
    mae=mean_absolute_error(Y_full_test,y_predict)
    rmse=np.sqrt(mean_squared_error(Y_full_test,y_predict))
    meanscore += score
    meanr2 += r2
    meanMAE+= mae
    meanRMSE += rmse


print("mean R2 score from 100 trial",meanscore/100)

print("mean MAE from 100 trial",meanMAE/100)

print("mean RMSE from 100 trial",meanRMSE/100)

# print(model.feature_importances_)
# viz = FeatureImportances(reg)
# viz.fit(X_full_test,Y_full_test)
# viz.show()


fig = plt.figure(figsize=(10,7))
sns.regplot(Y_full_test, y_predict, color='blue', marker='+')
plt.show()


a = y_predict
b = y_predict

print(ttest_ind(a,b))

### You can see that after comparing the t statistic with the critical t value (computed internally) we get a good p value of 0.0005 and thus we reject the null hypothesis and thus it proves that the mean of the two distributions are different and statistically significant.


# ## Cross Checking with the internal scipy function
# t2, p2 = stats.ttest_ind(a,b)
# print("t = " + str(t2))
# print("p = " + str(p2))

# plt.rcdefaults()
# fig, ax = plt.subplots()
# people = ('Engine Size', 'Year', 'Model', 'Transmission', 'Fuel Type', 'Tax', 'Mileage', 'MPG')
# y_pos = np.arange(len(people))
# performance = [100,78,41,17,-1,-30,-38,-44]
# color = ['#77AC30','#0072BD','#4DBEEE','#EDB120','#7E2F8E','#A2142F','#77AC30','#0072BD']
#
# ax.barh(y_pos, performance, align='center', color=color)
# ax.set_yticks(y_pos)
# ax.set_yticklabels(people)
# ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xlabel('Relative Importances')
# ax.set_title('Feature Importances of 8 Features using DecisionTreeRegressor')
#
# plt.show()
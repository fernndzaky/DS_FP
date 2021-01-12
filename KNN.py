from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

# from yellowbrick.model_selection import FeatureImportances



df = pd.read_csv('audi.csv')
# pd.set_option('display.max_columns',9)
#REPLACING STRINGS INTO ZEROS AND ON ES
df['transmission'].replace(['Manual','Automatic','Semi-Auto'],['0','1','2'], inplace=True)
df['fuelType'].replace(['Petrol','Diesel','Hybrid'],['0','1','2'], inplace=True)
df['model'].replace([' A1',' A6',' A4',' A3',' Q3',' Q5',' A5',' S4',' Q2',' A7',' TT',' Q7',' RS6',' RS3',' A8',' Q8',' RS4',' RS5',' R8',' SQ5',' S8',' SQ7',' S3',' S5',' A2',' RS7'],['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'], inplace=True)

# print(df.info())
X = df[['model','year','transmission','mileage','fuelType','tax','mpg','engineSize']]
Y = df[['price']]

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


rmse_val = [] #to store rmse values for different k

for K in range(20):
    K +=1 #increment k value
    model = neighbors.KNeighborsRegressor(n_neighbors=K) #initiate KNN model
    model.fit(X_full_train, Y_full_train) #fit the train data set to model
    pred = model.predict(X_full_test) #predict using test dataset
    error = sqrt(mean_squared_error(Y_full_test, pred)) #calculate rmse
    rmse_val.append(error) #store into array for plotting
    print('RMSE alue for k=', K, ' is: ', error)

    # print('model score', model.score(X_full_test,Y_full_test))
    r2= r2_score(Y_full_test,pred)
    mae=mean_absolute_error(Y_full_test,pred)
    print('r2 score is', r2)
    print('mae score is', mae)



    # df = pd.DataFrame({'Actual':Y_full_test, 'Predicted':pred.astype(int)})
    # print(df)

# print(rmse_val)
# # print(re)
# # print('printing elbow curve')
# curve = pd.DataFrame(rmse_val) #elbow curve
# print(curve)
# curve.plot()
n = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
fig = plt.figure(figsize=(10,7))
plt.bar(n,rmse_val)
plt.title('K neighbors with its Mean Squared Error value')
plt.xlabel('n K neighbor')
plt.ylabel('Mean Squared Error')
plt.show()

meanr2 = 0
meanscore = 0
meanMAE = 0
meanRMSE = 0
for i in range(100):
    model = neighbors.KNeighborsRegressor(n_neighbors=3)  # initiate KNN model
    model.fit(X_full_train, Y_full_train)  # fit the train data set to model
    pred = model.predict(X_full_test)  # predict using test dataset
    error = sqrt(mean_squared_error(Y_full_test, pred))  # calculate rmse
    rmse_val.append(error)  # store into array for plotting
    # model.score(X_full_test, Y_full_test)
    r2 = r2_score(Y_full_test,pred)
    mae = mean_absolute_error(Y_full_test,pred)
    meanr2 += r2
    meanMAE += mae
    meanRMSE += error

# iteration = []
# for i in range (len(pred)):
#     iteration.append(i)
#
# plt.plot(iteration[0:100], pred[0:100], label="Predicted")
# plt.plot(iteration[0:100], Y_full_test[0:100].flatten(), label="Actual")
# plt.legend()
# plt.title("Actual vs Prediction for Decision KNN Regression")
# plt.show()

print('mean r2 score after 100 trial=', meanr2 / 100)
print('mean MAE score after 100 trial=', meanMAE / 100)
print('mean RMSE score after 100 trial', meanRMSE / 100)


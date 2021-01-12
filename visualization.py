import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('audi.csv')

# X = df[['model','year','transmission','mileage','fuelType','tax','mpg','engineSize']]
X = df['model']
Y = df['price']
label_size = 10
plt.rcParams['xtick.labelsize'] = label_size
plt.rc('axes', labelsize=label_size)

fig = plt.figure(figsize=(10,7))
plt.scatter(X,Y)
plt.xlabel('Car Model')
plt.ylabel('Car Price (in euro)')
plt.title('Car Price and Model Comparison')

X = df['year']
label_size = 14
plt.rcParams['xtick.labelsize'] = label_size
plt.rc('axes', labelsize=label_size)
fig = plt.figure(figsize=(10,7))
plt.scatter(X,Y)
plt.xlabel('Year Bought')
plt.ylabel('Car Price')
plt.title('Car Price and Year Bought Comparison')

X = df['transmission']
label_size = 14
plt.rcParams['xtick.labelsize'] = label_size
plt.rc('axes', labelsize=label_size)
fig = plt.figure(figsize=(10,7))
plt.scatter(X,Y)
plt.xlabel('Transmission')
plt.ylabel('Car Price')
plt.title('Car Price and Transmission Comparison')

X = df['mileage']
label_size = 14
plt.rcParams['xtick.labelsize'] = label_size
plt.rc('axes', labelsize=label_size)
fig = plt.figure(figsize=(10,7))
plt.scatter(X,Y)
plt.xlabel('Car Mileage')
plt.ylabel('Car Price')
plt.title('Car Price and Mileage Comparison')

X = df['fuelType']
label_size = 14
plt.rcParams['xtick.labelsize'] = label_size
plt.rc('axes', labelsize=label_size)
fig = plt.figure(figsize=(10,7))
plt.scatter(X,Y)
plt.xlabel('Fuel Type')
plt.ylabel('Car Price')
plt.title('Car Price and Fuel Type Comparison')

X = df['tax']
label_size = 14
plt.rcParams['xtick.labelsize'] = label_size
plt.rc('axes', labelsize=label_size)
fig = plt.figure(figsize=(10,7))
plt.scatter(X,Y)
plt.xlabel('Car Tax')
plt.ylabel('Car Price')
plt.title('Car Price and Tax Comparison')

X = df['mpg']
label_size = 14
plt.rcParams['xtick.labelsize'] = label_size
plt.rc('axes', labelsize=label_size)
fig = plt.figure(figsize=(10,7))
plt.scatter(X,Y)
plt.xlabel('Miles per Gallon')
plt.ylabel('Car Price')
plt.title('Car Price and Miles per Gallon Comparison')

X = df['engineSize']
label_size = 14
plt.rcParams['xtick.labelsize'] = label_size
plt.rc('axes', labelsize=label_size)
fig = plt.figure(figsize=(10,7))
plt.scatter(X,Y)
plt.xlabel('Engine Size (in litres)')
plt.ylabel('Car Price')
plt.title('Car Price and Engine Size Comparison')

X = df['engineSize']
Y = df['mpg']
label_size = 14
plt.rcParams['xtick.labelsize'] = label_size
plt.rc('axes', labelsize=label_size)
fig = plt.figure(figsize=(10,7))
plt.scatter(X,Y)
plt.xlabel('Engine Size (in litres)')
plt.ylabel('Miles Per Gallon')
plt.title('Miles Per Gallon and Engine Size Comparison')

# barWidth = 0.25
# fig = plt.subplots(figsize =(12, 8))
# # set height of bar
# engineSize = df['engineSize']
# mpg = df['mpg']
# #IT = [mean_valence_2015, mean_valence_2016, mean_valence_2017, mean_valence_2018, mean_valence_2019, mean_valence_2020]
# #ECE = [mean_danceability_2015, mean_danceability_2016, mean_danceability_2017, mean_danceability_2018, mean_danceability_2019, mean_danceability_2020]
# #CSE = [mean_speechiness_2015, mean_speechiness_2016, mean_speechiness_2017, mean_speechiness_2018, mean_speechiness_2019, mean_speechiness_2020]
#
# # Set position of bar on X axis
# br1 = np.arange(len(engineSize))
# br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
#
# # Make the plot
# plt.bar(br1, engineSize, color ='r', width = barWidth,
#         edgecolor ='grey', label ='IT')
# plt.bar(br2, mpg, color ='g', width = barWidth,
#         edgecolor ='grey', label ='ECE')
#
# # Adding Xticks
# plt.xlabel('Year', fontweight ='bold')
# plt.ylabel('Average Number', fontweight ='bold')
# plt.xticks([r + barWidth for r in range(len(engineSize))],
#            engineSize)
# colors = {'valence':'red', 'danceability':'green', 'speechiness':'blue'}
# labels = list(colors.keys())
# handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
# plt.legend(handles, labels)
# plt.title("Average of Valence, Danceability and Speechiness in 5 Years")

plt.show()

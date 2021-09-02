#import of library..
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import numpy
from sklearn.model_selection import train_test_split

#read pandas of csv
df = pd.read_csv('Machine Learning Software for Estimating Average Wind Speed.csv')

#data :)
X = df.drop(['rowID','hpwren_timestamp','avg_wind_direction','avg_wind_speed','max_wind_direction','max_wind_speed','min_wind_direction','min_wind_speed','rain_accumulation','rain_duration','relative_humidity'],axis = 'columns').values

#target data :)
y = df.iloc[:,5:6].values

#data to numpy array
x_array = numpy.array(X)

#train of data
X_train, X_test, y_train, y_test = train_test_split(x_array,y,test_size=0.1,random_state=20)

#to model :)
k_means_clustering = KMeans(n_clusters = 5, random_state = 20)

#model of prediction :)
kmeans_fit= k_means_clustering.fit_predict(X_train,y_train)


air_pressure = input("Enter Air Pressure: ")
air_temp = input("Enter Air Temp: ")

#with data of prediction.
prediction_with_kmeans_cluster = k_means_clustering.predict([[air_pressure,air_temp]])
for prediction_with_kmeans_cluster in y:
    tolist = prediction_with_kmeans_cluster.tolist()
    print("Prediction Average Wind Speed: {} ".format(tolist[0]))
    break


#cluster of image..
plt.scatter(X_train[kmeans_fit == 0, 0], X_train[kmeans_fit == 0, 1], s = 100, c = 'pink', label = 'Cluster 1')
plt.scatter(X_train[kmeans_fit == 1, 0], X_train[kmeans_fit == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_train[kmeans_fit == 2, 0], X_train[kmeans_fit == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_train[kmeans_fit == 3, 0], X_train[kmeans_fit == 3, 1], s = 100, c = 'purple', label = 'Cluster 4')
plt.scatter(X_train[kmeans_fit == 4, 0], X_train[kmeans_fit == 4, 1], s = 100, c = 'brown', label = 'Cluster 5')
plt.scatter(k_means_clustering.cluster_centers_[:, 0], k_means_clustering.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Cluster Centers')
plt.title('Air Pressure and Temperature Measurement')
plt.xlabel('Air Pressure')
plt.ylabel('Air Temp')
plt.legend()
plt.show()

#end...

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

dftr=pd.read_csv('drive/My Drive/Sangam_2019_Hackathon_Data.csv',parse_dates=['svrtime'],index_col=['device_id','svrtime'])

df=pd.read_csv('drive/My Drive/Sangam_2019_Hackathon_Data.csv')

df.head()

dftr.head()

# from google.colab import drive
# drive.mount('/content/drive')

# tr_x=(df[df['timestamp'] != '0000-00-00 00:00:00'])

#Timestamp rectification
#############################
df.timestamp=df.timestamp.replace('0000-00-00 00:00:00', np.NaN)

df.timestamp=pd.to_datetime(df.timestamp)

df.shape[0]

yr=np.array([])

for i in range(df.shape[0]):
  yr=np.append(yr,df.timestamp[i].year)

for i in range(df.shape[0]):
  if(yr[i].year != 2019):
      df.loc[i,'timestamp']=np.NaN

df.loc[:,'timestamp']=df.loc[:,'timestamp'].fillna(method='ffill')

tr_x=dftr[~dftr.index.duplicated(keep='first')]
xx=tr_x.resample('M').interpolate(method='linear')

##################################


plt.plot(x0.ch4)

plt.plot(x0.h2)

plt.plot(x0.no2)

plt.plot(x0.co)

plt.plot(x0.so2_tmp)

plt.plot(x0.latitude)

plt.scatter(x0.index,x0.latitude)





#Latitude and Longitude Imputation
#####################################
from sklearn.preprocessing import Imputer

im=Imputer(strategy='most_frequent')

dftr.latitude=dftr.latitude.replace(0, np.NaN)
dftr.longitude=dftr.longitude.replace(0, np.NaN)

dftr.loc['S9','latitude']=13.088279
dftr.loc['S9','longitude']=80.181568

dftr.drop('nso',axis=1,inplace=True)

dftr.drop('ewo',axis=1,inplace=True)

dftr.drop('timestamp',axis=1,inplace=True)

# dftr=pd.get_dummies(dftr)

tr=im.fit_transform(dftr)

tr=pd.DataFrame(tr)
tr.columns=dftr.columns

tr.head()

tr.drop('S.no',axis=1,inplace=True)

###############################################



#Plots to detect outliers
#####################################
plt.plot(tr.humidity)

plt.plot(tr.temperature)

plt.plot(tr.heat_index)

plt.plot(tr.uv)

plt.plot(tr.ir)

plt.plot(tr.luminence)

plt.plot(tr.co)

plt.plot(tr.no2)

plt.plot(tr.nh3)

plt.plot(tr.h2)

plt.plot(tr.pm01)

plt.plot(tr.pm10)

plt.plot(tr.pm25)

plt.plot(tr.so2_tmp)

plt.plot(tr.so2_gas)

plt.plot(tr.c2h5o4)

tr[tr.c2h5o4>800]

#########################################




#Outliers Removal
##########################################
tr=tr[tr.temperature<100]

tr=tr[tr.no2<500]

tr=tr[tr.nh3<1000]

tr=tr[tr.humidity<200]

tr=tr[tr.h2<1000]

tr=tr[tr.so2_tmp<100]

tr=tr[tr.so2_conc<10000]


###########################################




#Neural Network models
########################
import tensorflow as tf
import keras
import keras.layers as layers

model2 = keras.Sequential([
    
    layers.Dense(64, input_shape=(32,)),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dropout(0.10),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(12, activation=tf.nn.relu),
    
    layers.Dense(12, activation=tf.nn.relu),
    layers.Dense(1)
  ])

xtrcnn=np.reshape(xtr.values,(xtr.values.shape[0],1,xtr.values.shape[1]))
ytrcnn=np.reshape(ytr.values,(ytr.shape[0],1))

model = keras.Sequential([
    layers.LSTM(64,input_shape=(xtrcnn.shape[1],xtrcnn.shape[2])),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dropout(0.10),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(12, activation=tf.nn.relu),
    
    layers.Dense(12, activation=tf.nn.relu),
    layers.Dense(1)
  ])

model2.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['mean_absolute_error', 'mean_squared_error'])

model2.fit(tr_x.values, tr_y, epochs=5)

model2.save('model.h5')

keras_file= "model.h5"


# Convert to TensorFlow Lite model for Android App.
##############################
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("android_model.tflite", "wb").write(tflite_model)




#Train set preparation for temporal prediction using Sarimax
#############################################
trx=tr.drop('pm25',axis=1)

ytr=tr.pm25

ind=dftr.reset_index().svrtime[:1048023]

ytr.index=ind

trx.index=ind

tr_x=trx[~trx.index.duplicated(keep='first')] 
tr_y=ytr[~ytr.index.duplicated(keep='first')]



#########################################

#pip install pyramid-arima


#auto-arima is used to get best parameters for SARIMAX(p,d,q)
#################################
from pyramid.arima import auto_arima

modelar = auto_arima(tr_y, trace=True, error_action='ignore', suppress_warnings=True)


#Best parameters obtained : p=1, q=1, d=2 , seasonal_order=(0,0,0,1)


##########################################




#pip install geopy

#Geopy is used to convert coordinates to street addresses
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="app")
location = geolocator.reverse("28.4578	 , 77.034") 
print(location.address)


# Street address of stationary stations
###################
#S1=Sector 11A, Gurugram, Haryana, 122001, India
#S3=Sector 37A, Dhulkot, Gurugram, Haryana, India
#S4=Sector 11A, Gurugram, Haryana, 122001, India
#S5=Sector 11A, Gurugram, Haryana, 122001, India
#S6=Ward 84, Zone 7 Ambattur, Chennai, Chennai district, Tamil Nadu, 600050, India
#S7=Ward 84, Zone 7 Ambattur, Chennai, Chennai district, Tamil Nadu, 600050, India
#S8=Arulmigu Vinaitheertha Vinayagar Temple, Ambattur Estate Rd, Ward 89, Zone 7 Ambattur, Chennai, Chennai district, Tamil Nadu, 600101, India
#S9=Ambattur Estate Rd, Ward 89, Zone 7 Ambattur, Chennai, Chennai district, Tamil Nadu, 600101, India
####################





#Visualizations
########################
import seaborn as sns

import matplotlib.pyplot as plt

plt.subplots(figsize=(12,10))
sns.barplot(data=tr_x.iloc[:,25:28],palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Value', fontsize=15)
plt.xticks(rotation=45,fontsize=10)
plt.xlabel('PM', fontsize=15)
plt.title('Pollution level',fontsize=24)
plt.savefig('sources_per_country_count.png')
plt.show()

plt.subplots(figsize=(8,8))
sns.barplot(data=tr_x.iloc[:,18:21],palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Value', fontsize=15)
plt.xticks(rotation=45,fontsize=10)
plt.xlabel('PM', fontsize=15)
plt.title('Pollution level',fontsize=24)
plt.savefig('sources_per_country_count.png')
plt.show()

corr = tr_x.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(10, 8))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=False, linewidths=.5, cbar_kws={"shrink": .5});



#Map to show stationary stations position
######################
import folium
map_osm = folium.Map(location=[13.088279, 80.181568],	#"13.088279,80.181568") 
                 
                     zoom_start=12) 

# for point in range(0, len(locationlist)):
folium.Marker([13.088279, 80.181568], popup='sta').add_to(map_osm)
folium.Marker([13.1046		 ,80.1710], popup='sta2').add_to(map_osm)
folium.Marker([13.005		 ,80.2398], popup='sta3').add_to(map_osm)
folium.Marker([13.16		 ,80.260], popup='sta4').add_to(map_osm)
folium.Polygon([[13.15		 ,80.250],[13.18,80.280],[13.20 ,80.270],[13.20,80.290]])    
map_osm




colors = ["windows blue", "amber", "faded green", "dusty purple"]
sns.set(rc={"figure.figsize": (15,10), "axes.titlesize" : 18, "axes.labelsize" : 12, 
            "xtick.labelsize" : 14, "ytick.labelsize" : 14 })



#seasonal decompose to detect stationarity trends and seasonality
###########################
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

seasonal_decompose(tr_y, model='additive',freq=200).plot()
# print("Dickey–Fuller test: p=%f" % adfuller(tr_y)[1])




############################
plt.subplots(figsize=(15,7))
sns.barplot(data=tr_x.iloc[:,15:18],palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Value', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('SO2', fontsize=20)
plt.title('Average So2', fontsize=24)
plt.show()

g = sns.jointplot("h2", "no2", data = tr_x[100:10000], kind="reg")

g = sns.jointplot("temperature", "humidity", data = tr_x[100:5000], kind="reg")

cols = ['no2','h2','pm01']
sns.pairplot(tr_x[cols][10:1000])

# dftr.loc['M1'].longitude=dftr.loc['M1'].longitude.fillna(method='ffill')
# dftr.loc['M2'].longitude=dftr.loc['M2'].longitude.fillna(method='ffill')
# dftr.loc['M3'].longitude=dftr.loc['M3'].longitude.fillna(method='ffill')
# dftr.loc['M4'].longitude=dftr.loc['M4'].longitude.fillna(method='ffill')
# dftr.loc['M5'].longitude=dftr.loc['M5'].longitude.fillna(method='ffill')
# dftr.loc['M6'].longitude=dftr.loc['M6'].longitude.fillna(method='ffill')
# dftr.loc['M7'].longitude=dftr.loc['M7'].longitude.fillna(method='ffill')

tr_x.head()

s7=dftr.loc['S7']['nh3']
s8=dftr.loc['S8']['nh3']
s9=dftr.loc['S9']['nh3']

plt.subplots(figsize=(10,7))
sns.barplot(data=[s7,s8,s9],palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Value', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('Stations', fontsize=20)
plt.title('NH3',fontsize=24)
plt.show()

# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(data=tr_x[1:10000])
fig.axis(ymin=0, ymax=100);
plt.xticks(rotation=90);

tr_x.head()

tr_x.shape
########################################




############################
#Calculation of Solar Intensity
############################
time=np.array([])

tt=tr_x.reset_index().svrtime

tt

#### Variables
# H:hour angle
# Z:Zenith angle
# X: latitude
# Y: solar declination angle
# SI: solar intensity

#Formula used
#H=15*(hour-12)
#Y=23.45*sin(360*(284+day)/365)
#Z=sin(x)sin(Y)+cos(x)cos(Y)cos(H)
#SI=1000*cos(Z)

for i in range(103756):
  time=np.append(time,pd.Timestamp(tt.values[i]))

hour=np.array([])

day=np.array([])

for i in range(103756):
  hour=np.append(hour,time[i].hour)

for i in range(103756):
  day=np.append(day,time[i].dayofyear)

H=15*(hour-12)

Y=23.45*np.sin(360*(284+day)/365)

X=tr_x.latitude

Z=np.sin(X)*np.sin(Y)+np.cos(X)*np.cos(Y)*np.cos(H)

SI=1000*np.cos(Z)

plt.scatter(SI,H)

g = sns.jointplot(SI,H, kind="reg")

g = sns.jointplot(SI,tr_x.latitude, kind="reg")

g = plt.plot(tt[100:1000],SI[100:1000])



#Time series prediction (Temporal prediction)
###########################
import statsmodels.api as sm

tsmodel=sm.tsa.statespace.SARIMAX(tr_y,exog=tr_x.values,order=(1,1,2),seasonal_order=(0,0,0,0))

ts=tsmodel.fit()

ts.predict(tr_y.index.values[0],tr_y.index.values[0],exog=tr_x.iloc[0].values)

ts.forecast(1000,exog=tr_x.iloc[0:1000])



#Spatio-Temporal Graphs
###########################

def generateBaseMap(default_location=[13.088279, 80.181568], default_zoom_start=12):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map

base_map=generateBaseMap()



from folium.plugins import HeatMap
# df_copy = df[df.month>4].copy()
# df_copy['count'] = 1
base_map = generateBaseMap()
HeatMap(data=tr_x[['latitude', 'longitude', 'temperature']].groupby(['latitude', 'longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map)

base_map

base_map_p01 = generateBaseMap()

HeatMap(data=tr_x[['latitude', 'longitude', 'pm01']].groupby(['latitude', 'longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map_p01)

base_map_p01

base_map_p25 = generateBaseMap()

# base_map_no2 = generateBaseMap()
HeatMap(data=tr_x[['latitude', 'longitude', 'pm10']].groupby(['latitude', 'longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map_p25)

base_map_p25

base_map_so2_gas = generateBaseMap()

# base_map_no2 = generateBaseMap()
HeatMap(data=tr_x[['latitude', 'longitude', 'so2_gas']].groupby(['latitude', 'longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map_so2_gas)

base_map_so2_gas


#Road condition evaluation
#####################################
dfro=dftr.loc[:,["accx","accy","accz","acctemp","gyrox","gyroy","gyroz"]]

dfm1=dfro.loc['M1']

plt.plot(dfro.accx[1:10000],dfro.gyrox[1:10000])

plt.plot(dfro.accx,dfro.acctemp)

sns.jointplot(dfro.accx,dfro.gyrox)

# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(data=dfro)
fig.axis(ymin=0, ymax=100);
plt.xticks(rotation=90);

plt.scatter(dfro.accx,dfro.gyrox)

plt.scatter(dfro.accz,dfro.gyroz)


###############################
#Spatial AQI chart in android app.

#Spatial estimation in android app.



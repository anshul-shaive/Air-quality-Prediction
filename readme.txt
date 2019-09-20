The given dataset was based on time-series pollution data collected from some stationary stations and some moving sensors fitted in vehicles.

Several values in the server time and timestamp time-series was repeated because they belong to different devices, so i created a MultiIndex for the dataset with first index being the device_id and second being the server time. This way the index was unique and the data from different devices were easy to explore.

When the devices is not fixed it throws incorrect data, so the first task was to correct the data( timestamp rectification). 
What i noticed was the incorrect data often corresponds with timestamps of years such as 1980 or 2035.
We knew that the data was collected in the year 2019, so i changed the values of those timestamps which does not have 2019 in them to Nan, which can the be filled with forward values of the same devices timestamps.

Next the latitude and longitude has missing values(denoted by 0) or incorrect values in case of gps being not fixed.
So i imputed the values corresponding to each device.

Next I plotted several graphs to detect the outliers in the dataset and removed them.

Then i used several plotting and charting libraries to build vizualizations including matplotlib, seaborn and folium.

Folium library was used to build spatio-temporal hotspots of city. 

I also include many map based graphs showing so2 concentration in different parts of city.

I developed an lstm model for time series prediction and converted that model with TensorFlowLiteConverter to a flat buffer (.tflite file), which then i integrated into an android app.

I calculated solar intensity from the data with the formula ( described in python script) and vizualized the results.

For time-series (temporal) prediction i used SARIMAX( Seasonal Auto Regressive Integrated Moving Average model).
I used auto_arima from pyramid library to obtain the best parameters for SARIMAX.
SARIMAX was used from stastsmodels library with the obtained parameters.

I then used the fitted model to forecast the pollution values for the next 1000 steps.

I created a different dataframe for analyzing the road condition from accelerometer and gyroscope readings.
I plotted several graphs and get insights such as:
1. the gyroscope and accelerometer reading along same axis has a positive corelation
2. from graphs certain points can be seen as bumps in the road where the values of meters fluctuate suddenly and return to normal after some time.
3. the temperature in the accelorometer increses suddenly as speed is suddenly decreased or some hindrance has occured on road.

Rest is covered in the android application which uses data from central board of pollution control.









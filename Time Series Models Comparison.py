from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import keras

data = pd.read_csv('C:/Time_Series/FB.csv')
data.head()

data.boxplot()

data.isnull().sum()

train_data, test_data = data[0:int(len(data)*0.8)], data[int(len(data)*0.8):]

plt.figure(figsize=(12,7))
plt.title('Apple Prices')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.plot(data['Close'], 'blue', label='Training Data')
plt.plot(test_data['Close'], 'green', label='Testing Data')
plt.legend()

#ARIMA
from statsmodels.tsa.arima_model import ARIMA

train_ar = train_data['Close'].values
test_ar = test_data['Close'].values

history = [x for x in train_ar]
# print(type(history))
predictions = list()

for t in range(len(test_ar)):
    model = ARIMA(history, order=(2,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)
    
plt.figure(figsize=(20,8))
plt.plot(test_ar)
plt.plot(predictions, color='red')
plt.title('Predictions Vs True values')
plt.ylabel('Closing Stock Price')
plt.xlabel('Time steps')
plt.legend(['True values', 'Predictions'], loc='best')

plt.figure(figsize=(20,8))
plt.plot(test_ar[:50])
plt.plot(predictions[:50], color='red')
plt.title('Predictions Vs True values (for only 50 time steps)')
plt.ylabel('Closing Stock Price')
plt.xlabel('Time steps')
plt.legend(['True values', 'Predictions'], loc='best')    
   
from sklearn.metrics import mean_squared_error
arima_mse = mean_squared_error(test_ar, predictions)
print('Testing Mean Squared Error: ', arima_mse)

from sklearn.metrics import mean_absolute_error
arima_mae = mean_absolute_error(test_ar, predictions)
print('Testing Mean Absolute Squared Error: ', arima_mae)

print(model_fit.summary())
  
#SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
train_sar = train_data['Close'].values
test_sar = test_data['Close'].values

history = [x for x in train_sar]
print(type(history))
predictions_sar = list()
for t in range(len(test_sar)):
    #model_sar = SARIMAX(history, order=((0,1,3)), seasonal_order = (0,1,1,4))
    model_sar = SARIMAX(history, order=((2,1,0)), seasonal_order = (2,1,0,3))
    model_fit_sar = model_sar.fit(disp=0)
    output_sar = model_fit_sar.forecast()
    yhat_sar = output_sar[0]
    predictions_sar.append(yhat_sar)
    obs = test_sar[t]
    history.append(obs)
    print(t)
    
plt.figure(figsize=(12,8))
plt.plot(test_sar, label = 'Original Values')
plt.plot(pd.DataFrame(predictions_sar),label = 'Predicted Values',color='red')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Closing Share Price')
plt.title('Original and Predicted FB Share Price')    

plt.figure(figsize=(20,8))
plt.plot(test_sar[:50])
plt.plot(predictions[:50], color='red')
plt.title('Predictions Vs True values (for only 50 time steps)')
plt.ylabel('Closing Stock Price')
plt.xlabel('Time steps')
plt.legend(['True values', 'Predictions'], loc='best')

from sklearn.metrics import mean_squared_error,mean_absolute_error
mse_sar = mean_squared_error(test_sar,predictions_sar)
mae_sar= mean_absolute_error(test_sar, predictions_sar)
print('Testing Mean Absolute Error: %.3f' % mae_sar)
print('Testing Mean Squared Error: %.3f' % mse_sar)

print(model_fit_sar.summary())

#Simple Exponential Smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing as ses

train_ar = train_data['Close'].values
test_ar = test_data['Close'].values

history = [x for x in train_ar]
print(type(history))
predictions = list()

for t in range(len(test_ar)):
    model = ses(history)
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)
    
plt.figure(figsize=(20,8))
plt.plot(test_ar)
plt.plot(predictions, color='red')
plt.title('Predictions Vs True values')
plt.ylabel('Closing Stock Price')
plt.xlabel('Time steps')
plt.legend(['True values', 'Predictions'], loc='best')

plt.figure(figsize=(20,8))
plt.plot(test_ar[:50])
plt.plot(predictions[:50], color='red')
plt.title('Predictions Vs True values (for only 50 time steps)')
plt.ylabel('Closing Stock Price')
plt.xlabel('Time steps')
plt.legend(['True values', 'Predictions'], loc='best')    

ses_mse = mean_squared_error(test_ar, predictions)
print('Testing Mean Squared Error: ', ses_mse)

ses_mae = mean_absolute_error(test_ar, predictions)
print('Testing Mean Squared Error: ', ses_mae)

#Exponential Smoothing

from statsmodels.tsa.holtwinters import ExponentialSmoothing as es


history = [x for x in train_ar]
print(type(history))
predictions = list()

for t in range(len(test_ar)):
    model = es(history)
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)
    
plt.figure(figsize=(20,8))
plt.plot(test_ar)
plt.plot(predictions, color='red')
plt.title('Predictions Vs True values')
plt.ylabel('Closing Stock Price')
plt.xlabel('Time steps')
plt.legend(['True values', 'Predictions'], loc='best')

plt.figure(figsize=(20,8))
plt.plot(test_ar[:50])
plt.plot(predictions[:50], color='red')
plt.title('Predictions Vs True values (for only 50 time steps)')
plt.ylabel('Closing Stock Price')
plt.xlabel('Time steps')
plt.legend(['True values', 'Predictions'], loc='best')   

hses_mse = mean_squared_error(test_ar, predictions)
print('Testing Mean Squared Error: ', hses_mse)

hses_mae = mean_absolute_error(test_ar, predictions)
print('Testing Mean Squared Error: ', hses_mae) 

#Comparison of various Models
#Mean Squared Error
objects = ('ARIMA', 'SARIMA with lag order 2', 'Simple Exponential Smoothing', 'Exponential Smoothing')
scores = [arima_mse, mse_sar, ses_mse, hses_mse]
plt.figure(figsize=(14, 10))
plt.bar(objects, scores, align='center', alpha=0.5)
plt.xticks(objects)
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error of Different Models')
plt.grid()
plt.show()

#Mean Absolute Error
objects = ('ARIMA', 'SARIMA with lag order 2',  'Simple Exponential Smoothing', 'Exponential Smoothing')
scores = [arima_mae, mae_sar, ses_mae, hses_mae]
plt.figure(figsize=(14, 10))
plt.bar(objects, scores, align='center', alpha=0.5)
plt.xticks(objects)
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error of Different Models')
plt.grid()
plt.show()

#Conclusion: We can see that both Mean Squared Error and Mean Absolute Error of Simple Exponential Smoothing, Exponential Smoothing are same and less than rest. Even ARIMA is very much closer to the above two statistical models. SARIMA model with lag order of 2, sessional period 1 had higher error.
#Maybe we may achieve better results if we correctly pick suitable sessional period. Recent irregular variations may have disturbed the seasonality of the data.
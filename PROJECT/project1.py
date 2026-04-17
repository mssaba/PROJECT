import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


d=pd.read_csv('car_price_dataset.csv')
df=d.copy()
#1 no null values


#2 checking skewness and kurtosis
n=df.select_dtypes(include=['float64','int64']).columns
for i in n:
    print(f'{i}')
    print(df[i].skew())
    print(df[i].kurt())
# this particular data has normal skewness and kurtosis

# testing and training 
l=LabelEncoder()
df['Brand']=l.fit_transform(df['Brand'])
df['Fuel_Type']=l.fit_transform(df['Fuel_Type'])
df['Transmission']=l.fit_transform(df['Transmission'])

x=df.drop(['Price'],axis=1)
y=df['Price']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
L=LinearRegression()
L.fit(x_train,y_train)

p=L.predict(x_test)
print("MEAN=",mean_squared_error(y_test,p))

#ploting using matplot lib

pt.plot (d['Price'].head(10),d["Brand"].head(10),linestyle='-',color='k')
pt.xlabel('price')
pt.ylabel('brand')
pt.show()

#forecsting using the data
from statsmodels.tsa.statespace.sarimax import SARIMAX
df['Model_Year']=pd.to_datetime(df["Model_Year"])



model=SARIMAX(df['Price'],order=(10,1,1),seasonal_order=(1,1,1,12))
m=model.fit()

forecast= m.forecast(steps=12)
print(forecast)

f=pd.date_range(start=df.index[-1],periods=13,freq='YE')[1:]

o=pd.DataFrame({
    'DATE':f,
    "Forecast": forecast
})
print(o.info())

pt.figure(figsize=(10,5))
pt.plot(df['Price'], label="Actual Price")
pt.plot(o['Forecast'], label="Forecast", color='red')
pt.title("Price Forecast (Next 10 )")
pt.legend()
pt.show()

# o[['DATE','Forecast']].to_excel('froecast.xlsx',index=False)
# print("file saved")

#creating a sql data base

import pymysql as p

D=p.connect(
    host='localhost',
    user='root',
    password='2002'
)
a=D.cursor()
# a.execute("Create database forecast")
c=p.connect(
    host='localhost',
    user='root',
    passwd='2002',
    database='forecast'
)
b=c.cursor()

try:
    b.execute('CREATE TABLE price (date DATETIME, forecastprice FLOAT)')
except:
    pass

for i, j in zip(o['DATE'], o['Forecast']):
    b.execute(
        "INSERT INTO price (date, forecastprice) VALUES (%s, %s)",
        (str(i), float(j))
    )
c.commit()

b.execute('SELECT * FROM price')
print(b.fetchall())
import pandas as pd
import seaborn as sb
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pmdarima
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.colors
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import tsfel




#MAIN
size=(7,5)
plt.style.use('seaborn')
sb.set_style("darkgrid")
color_list = ['blue','red','green']
#cmap = matplotlib.colors.LinearSegmentedColormap.from_list(cluster_values, color_list)



#IMPORT
df=pd.read_csv('Data kategorier.csv',skiprows=2,index_col=['READ_TIME_Date'])
df.index = pd.to_datetime(df.index)


# 1) Weekday vs Weekend for apartments and houses
Houses=df.iloc[:,5:8]
Houses.columns=['sum','antal','average']
Houses['Type']='House'
Apartments=df.iloc[:,92:95]
Apartments.columns=['sum','antal','average']
Apartments['Type']='Apartment'
data=pd.concat([Houses,Apartments])
data['dayn']=data.index.weekday
data['Day']=data.dayn.apply(lambda x: 'weekend' if x>=5 else 'weekday')
plt.figure(figsize=size)
ax=sb.lineplot(data.index.hour,data.average,hue=data.Type,style=data.Day)
ax.set(xlabel='Hour', ylabel='Average household consumption [kW]')
plt.xlabel=('Hour')
plt.ylabel=('Average household consumption [kW]')
plt.show()


# 2) Age over months
A1 = df.loc[:,df.columns.isin(k for k in df.columns if 'Average' in k and 'a1' in k)]
A1['Average']=A1.mean(axis=1)
A1['Age']='18-30'
A2 = df.loc[:,df.columns.isin(k for k in df.columns if 'Average' in k and 'a2' in k)]
A2['Average']=A2.mean(axis=1)
A2['Age']='30-65'
A3 = df.loc[:,df.columns.isin(k for k in df.columns if 'Average' in k and 'a3' in k)]
A3['Average']=A3.mean(axis=1)
A3['Age']='65+'
data=pd.concat([A1.iloc[:,-2:],A2.iloc[:,-2:],A3.iloc[:,-2:]])
data['Month']=data.index.month_name(locale='French')
data['Monthn']=data.index.month
data['Season']=data.Monthn.apply(lambda x: 'Winter' if x in [12,1,2] else('Spring' if x in [3,4,5] else('Summer' if x in [6,7,8] else ('Fall'))))
plt.figure(figsize=size)
ax=sb.lineplot(data.index.hour,data.Average,hue=data.Age,style=data.Season, ci=None)
ax.set(xlabel='Hour', ylabel='Average household consumption [kW]')
plt.show()




# 3) Number of children vs Weekend and age 30-64

C0 = df.loc[:,df.columns.isin(k for k in df.columns if 'Average' in k and '0b' in k and 'a2' in k)]
C0['Average']=C0.mean(axis=1)
C0['Children']='No'
C1 = df.loc[:,df.columns.isin(k for k in df.columns if 'Average' in k and 'mb' in k and 'a2' in k)]
C1['Average']=C1.mean(axis=1)
C1['Children']='Yes'
data=pd.concat([C0.iloc[:,-2:],C1.iloc[:,-2:]])
data['dayn']=data.index.weekday
data['Day']=data.dayn.apply(lambda x: 'weekend' if x>=5 else 'weekday')
plt.figure(figsize=(10,5))
ax = plt.subplot(122)
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ax = sb.lineplot(data.index.hour,data.Average,hue=data.Children,style=data.Day, ci=None)
ax.set(xlabel='Hour', ylabel='')
plt.axvline(x=18)
ax.title.set_text('Age: 30-64')
plt.show()

#Age 18-30
C0 = df.loc[:,df.columns.isin(k for k in df.columns if 'Average' in k and '0b' in k and 'a1' in k)]
C0['Average']=C0.mean(axis=1)
C0['Children']='No'
C1 = df.loc[:,df.columns.isin(k for k in df.columns if 'Average' in k and 'mb' in k and 'a1' in k)]
C1['Average']=C1.mean(axis=1)
C1['Children']='Yes'
data1=pd.concat([C0.iloc[:,-2:],C1.iloc[:,-2:]])
data1['dayn']=data1.index.weekday
data1['Day']=data1.dayn.apply(lambda x: 'weekend' if x>=5 else 'weekday')
ax1 = plt.subplot(121, sharey = ax)
ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
ax1 = sb.lineplot(data1.index.hour,data1.Average,hue=data1.Children,style=data1.Day, ci=None)
ax1.set(xlabel='Hour', ylabel='Average household consumption [kW]')
plt.axvline(x=18)
ax1.title.set_text('Age: 18-30')
plt.show()


#Number of adults


H1 = df.loc[:,df.columns.isin(k for k in df.columns if 'Average' in k and 'a1' in k)]
H1['Average']=C0.mean(axis=1)
H1['Adults']='One'
H2=df.loc[:,df.columns.isin(k for k in df.columns if 'Average' in k and 'h2' in k)]
H2['Average']=C1.mean(axis=1)
H2['Adults']='Two'
data=pd.concat([H1,H2])
data['dayn']=data.index.weekday
data['Day']=data.dayn.apply(lambda x: 'weekend' if x>=5 else 'weekday')
plt.figure(figsize=size)
ax=sb.lineplot(data.index.hour,data.Average,hue=data.Adults,style=data.Day)
ax.set(xlabel='Hour', ylabel='Average household consumption [kW]')
plt.axvline(x=18)
plt.xlabel=('Hour')
plt.ylabel=('Average household consumption [kW]')
plt.show()










































#################TRASH



plt.figure(figsize=(10,10))
ax = plt.subplot(221)
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
age=data[data.Age=='18-30']
ax=sb.lineplot(age.index.hour,age.Average,hue=age.Month)
ax = plt.subplot(222)
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
age=data[data.Age=='30-65']
ax=sb.lineplot(age.index.hour,age.Average,hue=age.Month)
ax = plt.subplot(223)
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
age=data[data.Age=='65+']
ax=sb.lineplot(age.index.hour,age.Average,hue=age.Month)




#Age and months
A1 = df.loc[:,df.columns.isin(k for k in df.columns if 'Average' in k and 'a1' in k)]
A1['Average']=A1.mean(axis=1)
A1['Age']='18-30'
A2 = df.loc[:,df.columns.isin(k for k in df.columns if 'Average' in k and 'a2' in k)]
A2['Average']=A1.mean(axis=1)
A2['Age']='30-65'
A3 = df.loc[:,df.columns.isin(k for k in df.columns if 'Average' in k and 'a3' in k)]
A3['Average']=A1.mean(axis=1)
A3['Age']='65+'
A3=pd.concat([A3,l3])
data=pd.concat([A1.iloc[:,-2:],A2.iloc[:,-2:],A3.iloc[:,-2:]])
data['Month']=data.index.month_name
plt.figure(figsize=size)
ax=sb.lineplot(data.index.hour,data.average,hue=data.Type,style=data.Day)
ax.set(xlabel='Hour', ylabel='Average household consumption [kW]')
plt.xlabel=('Hour')
plt.ylabel=('Average household consumption [kW]')
plt.show()






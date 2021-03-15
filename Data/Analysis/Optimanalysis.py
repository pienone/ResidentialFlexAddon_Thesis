#%%
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import calendar
#from pylab import *

#%% IMPORT
FlexF1=pd.read_csv('FlexF1.csv',header=None)
df=pd.read_csv('Data kategorier.csv',skiprows=2,index_col=['READ_TIME_Date'])
df.index=pd.to_datetime(df.index)
FlexF1.index=df.index[0:8759]
prices=pd.read_csv('pricesDK1.csv')
EV=pd.read_csv('EVweek.csv')


#%%
months= np.arange(1,13)
plt.figure(figsize=(20,10))
for m in months:
    month = FlexF1[FlexF1.index.month==m]
    plt.subplot(3,4,m)
    ax=sb.lineplot(month.index.hour, month.iloc[:,1])
    ax.set_title('Month= %s'%calendar.month_name[m])
    ax.set_xlabel('Hour')
    ax.set_ylabel('Heating')

#%%EV
EV.index=df.index[0:EV.shape[0]]
EVuse=pd.read_csv('EVweek.csv')
p=prices.iloc[1176:1176+167,1]
p.index=EV.index
plt.figure()
ax=plt.subplot(1,2,1)
for type in EV.columns:
    plt.subplot(1,2,1)
    ax=sb.lineplot(EV.index, EV[type])
    ax.set_xlabel('')
    plt.subplot(1,2,2)
    sb.lineplot(EV.index.hour,EV[type])
    ax.set_xlabel('Hour')
ax2 = ax.twinx()
ax2=sb.lineplot(p.index,p)
ax2.set_ylabel('Price')




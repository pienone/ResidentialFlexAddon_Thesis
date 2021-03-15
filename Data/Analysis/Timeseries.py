import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import pmdarima
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



# Import
df = pd.read_csv('Old/Data kategorier.csv', skiprows=2, index_col=['READ_TIME_Date'])
df.index = pd.to_datetime(df.index)
sb.set_theme(style="whitegrid")


# Main
Average = df.loc[:, df.columns.isin([k for k in df.columns if 'Average_' in k])]
Average.columns = [k.split("_")[1] for k in Average.columns]
Average = Average.drop('huser', axis=1)
Average = Average.drop('lejligheder', axis=1)
Categories = Average.columns

# Parameters calculation
summary = Average.describe().T
summary['Total [MWh]'] = Average.sum().T / 1000
summary['Peak'] = Average.max().T
summary['Average'] = Average.mean().T
summary['PeakToAverage'] = summary['max'] / summary['mean']
# Seasonal averages
summary['Winter AVG'] = Average[Average.index.month.isin([12, 1, 2])].mean().T
summary['Spring AVG'] = Average[Average.index.month.isin([3, 4, 5])].mean().T
summary['Summer AVG'] = Average[Average.index.month.isin([6, 7, 8])].mean().T
summary['Fall AVG'] = Average[Average.index.month.isin([9, 10, 11])].mean().T
# Save summary
summary.to_csv('Time series\Timeseries_summary.csv')
summary.to_latex(multirow=True)

# Parameters across classes
parameters = summary.dropna().describe().T[8:]
parameters['Max class'] = [summary.dropna()[k].idxmax(axis=0) for k in parameters.index]
parameters['Min class'] = [summary.dropna()[k].idxmin(axis=0) for k in parameters.index]
parameters.drop(['count', '25%', '50%', '75%'], axis=1).to_csv('Time series\paramcalsses.csv')
# Load duration curves for max and min
maxclass = parameters['Max class'].unique()
minclass = parameters['Min class'].unique()
for Classes in [minclass, maxclass]:
    plt.figure()
    for category in Classes:
        serie = Average[category].sort_values(ascending=False)
        serie.index = np.arange(0, serie.shape[0])
        sb.lineplot(serie.index, serie, label=category)
    plt.xlabel('Sorted year hours')
    plt.ylabel('Average power consumption [kWh]')
    plt.legend()
# colours  bordeaux=#aa0000ff    green= #1aa21aff


# Maximum days and minimum days
Maxdays = [Average[k].idxmax() for k in Average.columns]
countMaxdays = [[x, Maxdays.count(x)] for x in set(Maxdays)]
Totantals=Antal[Average.columns[list(Average[k].idxmax()==countMaxdays[21][0] for k in Average.columns)]].loc[countMaxdays[21][0],:]
Totantals=Antal[Average.columns[list(Average[k].idxmax()==countMaxdays[22][0] for k in Average.columns)]].loc[countMaxdays[22][0],:]
Totantals=Antal[Average.columns[list(Average[k].idxmax()==countMaxdays[12][0] for k in Average.columns)]].loc[countMaxdays[12][0],:]

Totantals=Antal[Average.columns[list(Average[k].idxmax()==countMaxdays[2][0] for k in Average.columns)]].loc[countMaxdays[2][0],:]
Totantals=Antal[Average.columns[list(Average[k].idxmax()==countMaxdays[8][0] for k in Average.columns)]].loc[countMaxdays[8][0],:]
Totantals.sum()
#list(Antal.loc[k[1],Categories[k[0]]] for k in list(enumerate(Maxdays))
Maxsize = Average.max(axis=0).values
Mindays = [Average[k].idxmin() for k in Average.columns]
Minsize = Average.min(axis=0).values
# calendar plot for peaks
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
#ax = sb.relplot(x=Maxdays, y=np.zeros(len(Maxdays)), size=Maxsize, color='Red', sizes=(5, 300)) #with size
ax = sb.relplot(x=list(set(Maxdays)), y=np.zeros(len(list(set(Maxdays)))), size=[l[1] for l in countMaxdays], color='Red', sizes=(50,300)) #with categ numbers
sb.set_style("ticks")
# plt.xlim([17166.0,17532.0])
ax.set(yticks=[0])
plt.tick_params(labelleft=False, left=False)
for a in ax.axes.flat:
    a.grid(True, axis='y')
sb.despine(left=True)
date_form = DateFormatter("%m/%y")
ax.ax.axes.xaxis.set_major_formatter(date_form)
ax.ax.axes.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
#plt.legend(bbox_to_anchor=(0, 0), loc="center right", frameon=False).set_title('       Peak\nconsumption \n       [kW]')
plt.legend(bbox_to_anchor=(0, 0), loc="center right", frameon=False).set_title('Number of\ncategories')
#plt.legend()

# Seasonal averages
Seasonals = pd.concat([pd.DataFrame({'Value': summary['Winter AVG'], 'Season': 'Winter'}),
                       pd.DataFrame({'Value': summary['Spring AVG'], 'Season': 'Spring'}),
                       pd.DataFrame({'Value': summary['Summer AVG'], 'Season': 'Summer'}),
                       pd.DataFrame({'Value': summary['Fall AVG'], 'Season': 'Fall'})])
sb.set_theme(style="whitegrid")
plt.figure()
sb.boxplot(x='Season', y='Value', data=Seasonals, palette="viridis")
plt.xlabel('')
plt.ylabel('Average consumption [kW]')
# Seasonal total
Seasonals = pd.concat(
    [pd.DataFrame({'Value': Average[Average.index.month.isin([12, 1, 2])].sum().T, 'Season': 'Winter'}),
     pd.DataFrame({'Value': Average[Average.index.month.isin([3, 4, 5])].sum().T, 'Season': 'Spring'}),
     pd.DataFrame({'Value': Average[Average.index.month.isin([6, 7, 8])].sum().T, 'Season': 'Summer'}),
     pd.DataFrame({'Value': Average[Average.index.month.isin([9, 10, 11])].sum().T, 'Season': 'Fall'})])
sb.set_theme(style="whitegrid")
plt.figure()
sb.boxplot(x='Season', y='Value', data=Seasonals)
plt.xlabel('')
plt.ylabel('Total consumption [kW]')


######### Linear models
LRdata = summary


# Dummies building
def dummy(data, condition):
    serie = pd.Series([condition in k for k in data.index]).astype(int).values
    return (serie)


LRdata['Children'] = dummy(LRdata, '0b')
LRdata['House'] = dummy(LRdata, 'h')
LRdata['Single'] = dummy(LRdata, '1v')
LRdata['Age2'] = dummy(LRdata, 'a2')
LRdata['Age3'] = dummy(LRdata, 'a3')
LRdata['EV'] = dummy(LRdata, 'mEV')
LRdata['HP'] = dummy(LRdata, '12') + dummy(LRdata, '22')

# Linear regression
LR = pd.DataFrame(index=LRdata.columns[8:-9].to_list(), columns=pd.MultiIndex.from_product(
    [list(LRdata.columns[-7:].values), ['pvalue int', 'pvalue coeff', 'R^2', 'LL', 'Coeff sign']]))
import statsmodels.api as sm

for param in LR.index:
    for variab in LR.columns.get_level_values(0):
        x = LRdata[variab].values
        y = LRdata[param].values
        x = sm.add_constant(x)
        model = sm.OLS(y, x, missing='drop').fit()
        LR.loc[param][variab, 'pvalue int'] = model.pvalues[0]
        LR.loc[param][variab, 'pvalue coeff'] = model.pvalues[1]
        LR.loc[param][variab, 'R^2'] = model.rsquared
        LR.loc[param][variab, 'LL'] = model.llf
        LR.loc[param][variab, 'nobs'] = model.nobs
        if model.pvalues[1] <= 0.05 and model.pvalues[1] > 0.001:
            LR.loc[param][variab, 'Coeff sign'] = 'Yes'
        elif model.pvalues[1] <= 0.001:
            LR.loc[param][variab, 'Coeff sign'] = 'Strong yes'
        else:
            LR.loc[param][variab, 'Coeff sign'] = 'No'

LR.head()
LR.to_latex()
LR.to_csv('Time series\Linearegressions.csv')
LR.loc[:, (slice(None), 'Coeff sign')].to_csv('Time series\LinearegressionsYES.csv')




########### Antals

Antal = df.loc[:, df.columns.isin([k for k in df.columns if 'Antal' in k])]
Antal.columns = [k.split("_")[1] for k in Antal.columns]
# Min-max antals for category
Antal = Antal.drop('huser', axis=1)
Antal = Antal.drop('lejligheder', axis=1)
Antals = Antal.describe().T
Antals['Max antals'] = Antal.max().T
Antals['Min antals'] = Antal.min().T
#Antals = Antals.drop('huser', axis=0)
#Antals = Antals.drop('lejligheder', axis=0)
Antals['Normstd'] = Antals['std'] / Antals['mean'] * 1000
Antals['Normstd'].idxmax(axis=0)
Antals['Normstd'].max(axis=0)
Antals['Dev %'] = (Antals['max'] - Antals['min']) / Antals['max']
Antals['Dev %'].max(axis=0)
Antals['Dev %'].idxmax(axis=0)
Antal[Antals['Dev %'].idxmax(axis=0)].idxmin(axis=0)
Antals['Hoursatmax'] = [sum(sum([Antal[k] == Antal[k].max(axis=0)])) for k in Antals.index]
Antals['Hoursatmax'].idxmax(axis=0)
Antals['Hoursatmax'].max(axis=0)
Antals['Hoursatmax'].idxmin(axis=0)
Antals['Hoursatmax'].min(axis=0)
totantal=Antal.max(axis=0).sum()
avavail=Antal.sum(axis=1)/totantal
Antal.sum(axis=1).mean()/totantal
totavail=Antal.sum(axis=1)/totantal

plt.figure()
ax=sb.lineplot(totavail[1:].index,totavail[1:])
date_form = DateFormatter("%m/%y")
ax.axes.xaxis.set_major_formatter(date_form)
ax.axes.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xlabel('')
plt.ylabel('Availability of smart meters')


Antal.iloc[:168,:].sum(axis=1).max()-Antal.iloc[:-168,:].sum(axis=1).max()

minantal=pd.concat([Antal.idxmin(),Antal.max()-Antal.min()],axis=1)
Antal.idxmin().describe()
minantal.plot()
set(Antal.sum(axis=1).sort_values()[:450].index.date)
sorted(Antal.idxmin().unique())
Antaldrop=[[Antal.index[k],sorted(Antal.iloc[k,:].sum()-Antal.iloc[k+1,:].sum()] for k in range(8759)]










########### TIME SERIES ###########

# decomposition
def decomposition(serie):
    decomposition = seasonal_decompose(serie.dropna(), model='moltiplicative', period=24)
    trend = decomposition.trend
    decomposition2 = seasonal_decompose(trend.dropna(), model='moltiplicative', period=168)
    trend2 = decomposition2.trend
    seasonal2 = decomposition2.seasonal
    residual2 = decomposition2.resid
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(221)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.plot(serie, label='Original', linewidth=0.5)
    ax = plt.subplot(222)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.plot(trend2, label='Trend', linewidth=0.5)
    ax = plt.subplot(223)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.plot(seasonal2, label='Seasonal', linewidth=0.2)
    ax = plt.subplot(224)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.plot(residual2, label='Residual', linewidth=0.5)
    plt.legend()
    return ax

trend=pd.DataFrame(columns=Categories)
for category in Categories[Average.sum()!=0]:
    serie = Average[category]
    decomposition = seasonal_decompose(serie.dropna(), model='moltiplicative', period=24)
    decomposition2 = seasonal_decompose(decomposition.trend.dropna(), model='moltiplicative', period=168)
    trend[category] = decomposition2.trend



EV=Average.loc[:, Average.columns.isin([k for k in Average.columns if 'mEV' in k])].mean(axis=1)
NOEV=Average.loc[:, Average.columns.isin([k for k in Average.columns if 'uEV' in k])].mean(axis=1)
APP=Average.loc[:, Average.columns.isin([k for k in Average.columns if 'A' in k])].mean(axis=1)
HOUSE=Average.loc[:, Average.columns.isin([k for k in Average.columns if 'h' in k])].mean(axis=1)
EH=Average.loc[:, Average.columns.isin([k for k in Average.columns if k[-1]=='2'])].mean(axis=1)
NOEH=Average.loc[:, Average.columns.isin([k for k in Average.columns if k[-1]=='1'])].mean(axis=1)


plt.figure()
for category in Categories[Average.sum()!=0]:
    if 'mEV' in category:
        sb.lineplot(trend.index, trend[category], color='#aa0000ff',alpha=0.4)
    if 'uEV' in category:
        sb.lineplot(trend.index, trend[category], color='#4c72b0ff',alpha=0.6)
#    plt.legend()
    plt.xlabel('')
    plt.ylabel('Average consumption [kW]')

plt.figure()
for category in Categories[Average.sum()!=0]:
    index=trend.index
    if category[-1]=='2':
        sb.lineplot(x=index, y=category, color='#aa0000ff',alpha=0.4,data=trend)
    if category[-1]=='1':
        sb.lineplot(x=index, y=category, color='#4c72b0ff',alpha=0.6,data=trend)
#    plt.legend()
    plt.xlabel('')
    plt.ylabel('Average consumption [kW]')




### WEEKLY

weeklytrend=pd.DataFrame(columns=Categories)
for category in Categories[Average.sum()!=0]:
    serie = Average[category]
    decomposition = seasonal_decompose(serie.dropna(), model='moltiplicative', period=24)
    #decomp2= seasonal_decompose(decomposition.trend.dropna(), model='moltiplicative', period=168)
    weeklytrend[category] = decomposition.trend.dropna()
weeklytrend=weeklytrend.iloc[24:,:]
weeklytrend['hour'] = pd.concat([pd.Series(np.arange(168))] * 53, axis=0).values[:8711]

plt.figure()
ax=sb.lineplot(weeklytrend.hour,weeklytrend.iloc[:,:-1].mean(axis=1),ci=25)
import matplotlib.ticker as ticker
# Hide major tick labels
ax.xaxis.set_major_locator(ticker.FixedLocator([0,24,48,72,96,120,144,168]))
ax.set_xticks([float(n)+0.5 for n in ax.get_xticks()])
ax.set_xticklabels('')
ax.xaxis.set_major_formatter(ticker.NullFormatter())
# Customize minor tick labels
ax.xaxis.set_minor_locator(ticker.FixedLocator([12,36,60,84,108,132,156]))
ax.xaxis.set_minor_formatter(ticker.FixedFormatter(['Mon','Tue','Wed','Thu','Fri','Sat','Sun']))
plt.xlabel('')
plt.ylabel('Average consumption deviation [kW]')


plt.figure()
for category in Categories[Average.sum()!=0]:
    for week in Average.index.week.unique()[1:]:
        series=weeklytrend.loc[weeklytrend.index.week==week,category]
        series.index=range(168)
        index=series.index
        sb.lineplot(x=index,y=series, color='#4c72b0ff',alpha=0.05)

plt.figure()
for category in Categories[Average.sum()!=0]:
    sb.lineplot(x=weeklytrend.hour,y=category, color='#4c72b0ff',alpha=0.3,data=weeklytrend)

plt.figure()
sb.lineplot(weeklytrend.hour,weeklytrend.iloc[:,:-1].mean(axis=1),ci=0)
plt.xlabel('Weekly hours')
plt.ylabel('Average consumption [kW]')


WEEKDAY=Average.loc[Average.index.weekday.isin([6,7])]
WEEKEND=Average.loc[Average.index.weekday.isin([1,2,3,4,5])]
plt.figure()
sb.lineplot(WEEKDAY.index.hour,WEEKDAY.mean(axis=1),color='#aa0000ff',ci=95,label='Weekend')
sb.lineplot(WEEKEND.index.hour,WEEKEND.mean(axis=1),ci=95,label='Weekday')
plt.legend(loc='upper left')
plt.xlabel('Hour of the day')
plt.ylabel('Average consumption [kW]  \n95% C.I.')

SINGLE=pd.DataFrame({'cons':SINGLEK})
SINGLE['weekday']='weekday'
SINGLE.loc[Average.index.weekday.isin([6,7]),'weekday']='weekend'
SINGLE['type']='single'
MULTI=pd.DataFrame({'cons':MULTIk})
MULTI['weekday']='weekday'
MULTI.loc[Average.index.weekday.isin([6,7]),'weekday']='weekend'
MULTI['type']='multi'
data=pd.concat([SINGLE,MULTI])




##################### DAILY ####################

Averageok=Average.loc[:,Average.sum()!=0]

# House type
APP=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if 'A' in k])].mean(axis=1).sum()
HOUSE=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if 'h' in k])].mean(axis=1).sum()
plt.figure()
sb.lineplot(APP.index.hour[1:],APP[1:],label='Apartments',color='#1aa21aff',ci=99)
sb.lineplot(HOUSE.index.hour[1:],HOUSE[1:],label='Houses',ci=99)
plt.xlabel('Hour')
plt.ylabel('Average consumption [kW]  \n99% C.I.')
plt.legend(loc='upper left')
APP.sum()-HOUSE.sum()
#AGE
a1=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if 'a1' in k])].mean(axis=1)
a2=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if 'a2' in k])].mean(axis=1)
a3=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if 'a3' in k])].mean(axis=1)
plt.figure()
sb.lineplot(a1.index.hour.values[1:],a1[1:],label='a1 < 30',ci=99)
sb.lineplot(a2.index.hour.values[1:],a2[1:],label='30 < a2 < 65',ci=99)
sb.lineplot(a3.index.hour.values[1:],a3[1:],label='65 < a3',ci=99)
plt.xlabel('Hour')
plt.ylabel('Average consumption [kW]  \n99% C.I.')
plt.legend(loc='upper left')

#Children
K=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if 'mb' in k])].mean(axis=1)
NOK=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if '0b' in k])].mean(axis=1)
plt.figure()
sb.lineplot(K.index.hour[1:],K[1:],label='With children',ci=99)
sb.lineplot(NOK.index.hour[1:],NOK[1:],label='Without children',ci=99)
plt.xlabel('Hour')
plt.ylabel('Average consumption [kW]  \n99% C.I.')
plt.legend(loc='upper left')


#Adults
SINGLE=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if '1v' in k])].mean(axis=1)
MULTI=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if '2v' in k])].mean(axis=1)
plt.figure()
ax=sb.lineplot(SINGLE.index.hour[1:],SINGLE[1:],label='Single',ci=99)
sb.lineplot(MULTI.index.hour[1:],MULTI[1:],label='Couple',ci=99)
plt.xlabel('Hour')
plt.ylabel('Average consumption [kW]  \n99% C.I.')
plt.legend(loc='upper left')



#subplots
import matplotlib.pyplot as plt
plt.figure()
ax1 = plt.subplot2grid(shape=(4,4), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((4,4), (0,2), colspan=2)
ax3 = plt.subplot2grid((4,4), (1,0), colspan=2)
ax4 = plt.subplot2grid((4,4), (1,2), colspan=2)
ax5 = plt.subplot2grid((4,4), (2,1), colspan=2)

ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)



ax1=sb.lineplot(APP.index.hour,APP,label='Apartments')
ax1=sb.lineplot(HOUSE.index.hour,HOUSE,label='Houses')
ax1.set(xlabel='Hour', ylabel='Average consumption [kW]  \n95% C.I.')
#plt.axvline(x=18)
#plt.axvline(x=17, c='#dd8452ff')
plt.legend(loc='upper left')

ax2=sb.lineplot(a1.index.hour.values,a1,label='a1 < 30')
ax2=sb.lineplot(a2.index.hour.values,a2,label='30 < a2 < 65')
ax2=sb.lineplot(a3.index.hour.values,a3,label='65 < a3')
ax2.set(xlabel='Hour', ylabel='Average consumption [kW]  \n95% C.I.')
ax3=sb.lineplot(K.index.hour,K,label='With children')
ax3=sb.lineplot(NOK.index.hour,NOK,label='Without children')
ax3=ax.set(xlabel='Hour', ylabel='Average consumption [kW]  \n95% C.I.')
#plt.axvline(x=18)
#plt.axvline(x=17, c='#dd8452ff')
plt.legend(loc='upper left')


plt.figure()




#EVNOEV
EV=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if 'mEV' in k])].mean(axis=1).sum()
NOEV=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if 'uEV' in k])].mean(axis=1).sum()
plt.figure()
sb.lineplot(EV.index.hour[24:],EV[24:],label='With EV')
sb.lineplot(NOEV.index.hour[24:],NOEV[24:],label='Without EV')
plt.xlabel('Hour')
plt.ylabel('Average consumption [kW]  \n95% C.I.')
#ax.set(xlabel='Hour', ylabel='Average consumption [kW]  \n95% C.I.')
#plt.axvline(x=18)
#plt.axvline(x=17, c='#dd8452ff')
plt.legend(loc='upper left')

#EHNOEH
EH=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if k[-1]=='2'])].mean(axis=1)
NOEH=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if k[-1]=='1'])].mean(axis=1)
plt.figure()
sb.lineplot(EH.index.hour[24:],EH[24:],label='With EH')
sb.lineplot(NOEH.index.hour[24:],NOEH[24:],label='Without EH')
plt.xlabel('Hour')
plt.ylabel('Average consumption [kW]  \n95% C.I.')
#ax.set(xlabel='Hour', ylabel='Average consumption [kW]  \n95% C.I.')
#plt.title()
#plt.axvline(x=18)
#plt.axvline(x=17, c='#dd8452ff')
plt.legend(loc='upper left')




EH=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if k[-1]=='2' and 'mE' not in k])].sum()
NOEH=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if k[-1]=='1'and 'mE' not in k])].sum()
EH.index=[k[:7] for k in EH.index]
NOEH.index=[k[:7] for k in NOEH.index]
NOEH=NOEH.loc[EH.index]

NOEH.loc[EH.index]
plt.figure()
plt.bar(EH.index,EH,color='#aa0000ff', label='Electric heating cons.',alpha=0.8)
plt.bar(EH.index,NOEH.loc[EH.index], label='Conventional cons.',alpha=0.8)
plt.ylabel('Total annual consumption [kWh]')
plt.xticks(rotation=60)
plt.legend()


NOEH.index.unique()



#Number of adults
H1 = df.loc[:,df.columns.isin(k for k in df.columns if 'Averageok' in k and 'a1' in k)]
H1['Averageok']=C0.mean(axis=1)
H1['Adults']='1'
H2=df.loc[:,df.columns.isin(k for k in df.columns if 'Averageok' in k and 'h2' in k)]
H2['Averageok']=C1.mean(axis=1)
H2['Adults']='2'
data=pd.concat([H1,H2])
data['dayn']=data.index.weekday
data['Day']=data.dayn.apply(lambda x: 'weekend' if x>=5 else 'weekday')
plt.figure(figsize=size)
ax=sb.lineplot(data.index.hour,data.Averageok,hue=data.Adults,style=data.Day,ci=95)
ax.set(xlabel='Hour', ylabel='Averageok consumption [kW]  \n95% C.I.')
plt.axvline(x=18)
plt.axvline(x=17, c='#dd8452ff')
plt.legend(loc='upper left')


















# ARIMA models

####### ARIMA
model = pmdarima.auto_arima(serie.dropna().T, start_p=1, start_q=1, max_p=3, max_q=3,
                            start_P=1, start_Q=1, max_P=3, max_Q=3, m=24,
                            seasonal=True, trace=True, d=1, D=1,
                            error_action='warn', suppress_warnings=True,
                            stepwise=True, random_state=20, n_fits=6)
model.summary()


model = ARIMA(Average.iloc[:, 1], order=(5,1,0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())





















# House and price correlation

price = pd.read_csv('pricesDK1.csv').dropna()
price.index = df.index[:-1]

houseprice = pd.concat([Average.loc[:, ['huser', 'lejligheder']], price['2020 DK1']], axis=1)
corrp = houseprice.groupby(Average.index.hour).corr()
corrp = corrp.iloc[corrp.index.get_level_values(1).isin(['2020 DK1']), :-1]
corrp.index = corrp.index.droplevel(1)
corrp.T.to_csv('correlationprice.csv')

corrp.groupby(corrp.index.get_level_values(1)).groupby(corrp.index.get_level_values(1))
correlation = pd.DataFrame(index=c)

# Duration curves
Ordered = Average
Ordered.index = np.arange(0, Ordered.shape[0])
plt.figure()
for category in Ordered.columns:
    serie = Ordered[category].sort_values(ascending=False)
    serie.index = np.arange(0, serie.shape[0])
    sb.lineplot(serie.index, serie, label=category)
plt.legend()

# Only houses and apartments
plt.figure()
for category in ['huser', 'lejligheder']:
    serie = Ordered[category].sort_values(ascending=False)
    serie.index = np.arange(0, serie.shape[0])
    ax = sb.lineplot(serie.index, serie, label=category)
ax.set_ylabel('Consumption [kW]')
ax.set_xlabel('Ordered year hours')
plt.legend()
import pandas as pd
import seaborn as sb
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pmdarima
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

size=(7,5)
plt.style.use('seaborn')

sb.set_theme(style="whitegrid")
color_list = ['blue','red','green']

#IMPORT

#df=pd.read_csv('Data kategorier.csv',skiprows=2,index_col=['READ_TIME_Date'])
#df.index = pd.to_datetime(df.index)

######## CLUSTERING DAILY ON ONE SERIE
serie=df.Average_h2vmba2uEV21

dfserie=pd.concat([df.Hour,serie],axis=1)
dfserie = dfserie.astype(np.float).fillna(method='bfill')
df_uci_hourly = dfserie.resample('H').sum()
df_uci_hourly.index = df_uci_hourly.index.date
df_uci_pivot = df_uci_hourly.pivot(columns='Hour')
df_uci_pivot = df_uci_pivot.Average_h2vmba2uEV21.dropna()


#df_uci_pivot.T.plot(figsize=(13,8), legend=False, color='blue', alpha=0.02)
sillhoute_scores = [0.3,0.3]
n_cluster_list = np.arange(2, 31).astype(int)

X = df_uci_pivot.values.copy()

# Very important to scale!
sc = MinMaxScaler()
X = sc.fit_transform(X)
df_uci_pivot
for n_cluster in n_cluster_list:
	kmeans = KMeans(n_clusters=n_cluster)
	cluster_found = kmeans.fit_predict(X)
	sillhoute_scores.append(silhouette_score(X, kmeans.labels_))
plt.figure()
ax = plt.plot(sillhoute_scores)


kmeans = KMeans(n_clusters=5)
cluster_found = kmeans.fit_predict(X)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
df_uci_pivot = df_uci_pivot.set_index(cluster_found_sr, append=True)

fig, ax= plt.subplots(1,1, figsize=(10,5))
color_list = ['blue','red','green','orange','black']
names=['week','weekend','holidays']
cluster_values = sorted(df_uci_pivot.index.get_level_values('cluster').unique())

for cluster, color in zip(cluster_values, color_list):
    df_uci_pivot.xs(cluster, level=1).T.plot(
        ax=ax, legend=False, alpha=0.1, color=color, label= f'Cluster {cluster}'
        )
    df_uci_pivot.xs(cluster, level=1).median().plot(
        ax=ax, color=color, alpha=0.9, ls='--',legend=False
    )

ax.set_xticks(np.arange(0,24))
ax.set_ylabel('Consumption [kW]')
ax.set_xlabel('Hour')
#plt.legend()
#ax.get_legend().remove()


from sklearn.manifold import TSNE
import matplotlib.colors

tsne = TSNE()
results_tsne = tsne.fit_transform(X)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(cluster_values, color_list)

plt.figure()
plt.scatter(results_tsne[:,0], results_tsne[:,1],
    c=df_uci_pivot.index.get_level_values('cluster'),
    cmap=cmap,
    alpha=0.6,
    )


#df_uci_pivot['week'] = pd.to_datetime(df_uci_pivot.index.get_level_values(0)).strftime('%W')
#df['week']=df.index.strftime('%W')
#dailymean=df_uci_pivot.iloc[0:-1].mean(axis=1)
#dailymean=pd.concat([dailymean[-50:],dailymean])
#dailymean=pd.concat([dailymean[365:],dailymean.dropna()])
#df_uci_pivot['rollingmean']=dailymean.dropna().rolling(window = 50).mean()
#df_uci_pivot['rollingmean']=dailymean[-50:50].dropna().rolling(window = 50).mean()


decomposition = seasonal_decompose(serie.dropna(),model='additive',period=24)
decomptrend=seasonal_decompose(decomposition.trend.dropna(),model='additive',period=168)
plt.figure()
ax = plt.subplot(111)
plt.scatter(pd.to_datetime(df_uci_pivot.index.get_level_values(0)),decomptrend.trend.resample('D').mean(),c=df_uci_pivot.index.get_level_values('cluster'), cmap=cmap,alpha=0.5)
ax.set_xlim((pd.to_datetime('01-01-2017'),pd.to_datetime('31-12-2017')))
plt.ylabel("Consumption [kW]")
ax1 = ax.twiny()
plt.scatter(pd.to_datetime(df_uci_pivot.index.get_level_values(0)),decomptrend.trend.resample('D').mean(),c=df_uci_pivot.index.get_level_values('cluster'), cmap=cmap,alpha=0.5)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%W"))
ax1.set_xlim((pd.to_datetime('01-01-2017'),pd.to_datetime('31-12-2017')))
ax1.set_xlabel('Week')
plt.show()

######## CLUSTERING MONTHLY ON ONE ALL SERIES

#monthlyall=df.loc[:, df.columns.isin(k for k in df.columns if 'Average')]

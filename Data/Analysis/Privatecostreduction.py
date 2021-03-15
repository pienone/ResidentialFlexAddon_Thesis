import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
scenario='Expected'      ### choose between Expected, RapidScenario and SlowScenario
iteration='4iter'               ### choose #it
adoption=0.25           ### choose



prices=pd.DataFrame(columns=pd.MultiIndex.from_product([['DK1','DK2'],years]))#,index=price.index)
for DK in ['DK1','DK2']:
    prices.loc[:,prices.columns.get_level_values(0)==DK]=pd.read_csv('Julia/prices/pricesweekly'+DK+scenario+str(adoption)+iteration+'.csv').values


evnumbers=pd.read_csv('JuliatoBALMOREL/evsharesfin_'+scenario+'.csv',header=[0,1,2],index_col=0)
evnumbers.iloc[0:12]=evnumbers.iloc[0:12]*0.8 #houses only 80% at home
evnumbers.iloc[12:24]=evnumbers.iloc[12:24]*0.1 #apartments only 10% at home
ehnumbers=pd.read_csv('JuliatoBALMOREL/ShareHP_categoriesfin_'+scenario+'.csv',header=[0,1,2],index_col=0)
categories=evnumbers.index
coefficients=pd.read_csv('JuliatoBALMOREL/evcoefficientstwist2.csv',skiprows=1,names=categories)/100 #[8*30]
AWDEM = pd.read_csv('Julia\DemAW.csv',skiprows=1, names=categories,index_col=[0]).fillna(0)
AWDEM.columns=categories[:-1]
AWDEM[categories[-1]]=0
AWDEM=pd.concat([AWDEM[1008:1176],AWDEM[2016:2184],AWDEM[4032:4200],AWDEM[6720:6888]])
AWDEMmean=AWDEM.mean(axis=1)
AADEM = pd.read_csv('Julia\DemAA.csv',skiprows=1, names=categories)
AADEM=pd.concat([AADEM[1008:1176],AADEM[2016:2184],AADEM[4032:4200],AADEM[6720:6888]])
AADEMmean=AADEM.mean(axis=1)
AASHDEM = pd.read_csv('Julia\DemSumHouse.csv',skiprows=1,names=categories)
AASHDEM=pd.concat([AASHDEM[1008:1176],AASHDEM[2016:2184],AASHDEM[4032:4200],AASHDEM[6720:6888]])
AASHDEMmean=AASHDEM.mean(axis=1)
BEV=pd.read_csv('Julia\FLEXNOBEV.csv',names=categories)
BEV=pd.concat([BEV[1008:1176],BEV[2016:2184],BEV[4032:4200],BEV[6720:6888]])
BEVmean=BEV.mean(axis=1)
PHEV=pd.read_csv('Julia\FLEXNOPHEV.csv',names=categories)
PHEV=pd.concat([PHEV[1008:1176],PHEV[2016:2184],PHEV[4032:4200],PHEV[6720:6888]])
PHEVmean=PHEV.mean(axis=1)

#### TECHNOLOGICAL SAVINGS ####
years=prices.columns.get_level_values(1)[:4]
noflhp=list(zip([AWDEMmean,AADEMmean,AASHDEMmean],['AW','AA','AASH']))
noflev=list(zip([BEVmean,PHEVmean],['BEV','PHEV']))
costs=pd.DataFrame(columns=pd.MultiIndex.from_product([years,['DK1','DK2'],['AW','AA','AASH','AWflex','AAflex','AASHflex','BEV','PHEV','BEVflex','PHEVflex']]))#(columns=)
totcosts=pd.DataFrame()#(columns=)
for y in years:
    for DK in ['DK1','DK2']:
        AWflex = pd.read_csv('JuliatoBALMOREL/Juliaout/FlexEH/FlexEHAW_' + DK + '_y' + y + '.csv', names=categories).iloc[:,:12].mean(axis=1)
        AAflex = pd.read_csv('JuliatoBALMOREL/Juliaout/FlexEH/FlexEHAA_' + DK + '_y' + y + '.csv', names=categories).mean(axis=1)
        AASHflex = pd.read_csv('JuliatoBALMOREL/Juliaout/FlexEH/FlexEHSumHouse_' + DK + '_y' + y + '.csv',names=categories).mean(axis=1)
        BEVflex = pd.read_csv('JuliatoBALMOREL/Juliaout/FlexEV/FlexEV_' + DK + '_y' + y + '.csv', header=None).mean(axis=1)
        PHEVflex = pd.read_csv('JuliatoBALMOREL/Juliaout/FlexEV/FlexPHEV_' + DK + '_y' + y + '.csv', header=None).mean(axis=1)
        flhp = list(zip([AWflex,AAflex,AASHflex],['AWflex','AAflex','AASHflex']))
        flev = list(zip([BEVflex, PHEVflex], ['BEVflex', 'PHEVflex']))
        for hp in noflhp:
            costs[y,DK,hp[1]]=prices[DK,y]*hp[0].values
        for fhp in flhp:
            costs[y,DK,fhp[1]]=prices[DK,y]*fhp[0].values
        for ev in noflev:
            costs[y, DK, ev[1]] = prices[DK,y] * ev[0].values
        for fev in flev:
            costs[y, DK, fev[1]] = prices[DK,y] * fev[0].values

totalcosts=costs.sum()#iloc[2200:].sum()
saving=pd.DataFrame(columns=pd.MultiIndex.from_product([years,['DK1','DK2']]),index=['AW','AA','AASH','BEV','PHEV'])
for y in years:
    for DK in ['DK1','DK2']:
        for tec in saving.index:
            saving[y,DK][tec]=(totalcosts[y,DK][tec]-totalcosts[y,DK][tec+'flex'])/totalcosts[y,DK][tec]
print(saving)
saving.to_csv('tecsavings.csv')



#SAVINGS for categories

df = pd.read_csv('Old/Data kategorier.csv', skiprows=2, index_col=['READ_TIME_Date'])
df.index = pd.to_datetime(df.index)
Average = df.loc[:, df.columns.isin([k for k in df.columns if 'Average_' in k])]
Average.columns = [k.split("_")[1] for k in Average.columns]
Averageok=Average.loc[:,Average.sum()!=0]
noevnoeh=Averageok.loc[:, Averageok.columns.isin([k for k in Averageok.columns if k[-1]=='1' and 'uEV' in k])]
old=pd.concat([noevnoeh[1008:1176],noevnoeh[2016:2184],noevnoeh[4032:4200],noevnoeh[6720:6888]])
costpost=pd.DataFrame(columns=pd.MultiIndex.from_product([years,['DK1','DK2'],['EVEH','EV','EH'],['pre','post','diff']]),index=categories)
for DK in ['DK1','DK2']:
    for year in years:
        AWprofile = pd.read_csv('JuliatoBALMOREL/Juliaout/FlexEH/FlexEHAW_' + DK + '_y' + year + '.csv', names=categories)
        AAprofile = pd.read_csv('JuliatoBALMOREL/Juliaout/FlexEH/FlexEHAA_' + DK + '_y' + year + '.csv', names=categories)
        AASHprofile = pd.read_csv('JuliatoBALMOREL/Juliaout/FlexEH/FlexEHSumHouse_' + DK + '_y' + year + '.csv',names=categories)
        for category in categories:
            chargingBEV = pd.read_csv('JuliatoBALMOREL/Juliaout/FlexEV/FlexEV_' + DK + '_y' + year + '.csv',header=None)
            chargingPHEV = pd.read_csv('JuliatoBALMOREL/Juliaout/FlexEV/FlexPHEV_' + DK + '_y' + year + '.csv',header=None)
            weights = [evnumbers.loc[category][(DK, 'BEV', year)], evnumbers.loc[category][(DK, 'PHEV', year)]]
            weightsEH= weights=[ehnumbers.loc[category][DK, 'AW', year],ehnumbers.loc[category][DK, 'AA', year],ehnumbers.loc[category][DK, 'AASH', year]]
            consBEV = sum(chargingBEV.iloc[:, i] * coefficients[category][i] for i in np.arange(20))
            consPHEV = sum(chargingPHEV.iloc[:, i] * coefficients[category][i + 20] for i in np.arange(10))
            flexev= (consBEV * weights[0] +consPHEV * weights[1])/sum(weights)
            flexheating= (AWprofile[category] * weightsEH[0] + AAprofile[category] * weightsEH[1] +AASHprofile[category] * weightsEH[2]) / sum(weights)
            costpost[year,DK,'EVEH','post'][category]=((flexev.values+flexheating.values+old[category].values)*prices[DK,year]).sum()
            costpost[year,DK,'EV','post'][category]=((flexev.values+old[category].values)*prices[DK,year]).sum()
            costpost[year,DK,'EH','post'][category]=((flexheating.values+old[category].values)*prices[DK,year]).sum()
            noconsBEV = sum(BEV.iloc[:, i] * coefficients[category][i] for i in np.arange(20))
            noconsPHEV = sum(PHEV.iloc[:, i] * coefficients[category][i + 20] for i in np.arange(10))
            noflexev = (noconsBEV * weights[0] + noconsPHEV * weights[1]) / sum(weights)
            noflexheating= ((AWDEM[category] * weightsEH[0]).values + (AADEM[category] * weightsEH[1]).values +(AASHDEM[category] * weightsEH[2]).values) / sum(weights)
            costpost[year, DK, 'EVEH', 'pre'][category] = ((noflexev + noflexheating+old[category].values).values * prices[DK, year]).sum()
            costpost[year, DK, 'EV', 'pre'][category] = ((noflexev.values+old[category].values) * prices[DK, year]).sum()
            costpost[year, DK, 'EH', 'pre'][category] = ((noflexheating+old[category].values) * prices[DK, year]).sum()
            costpost[year, DK, 'EVEH', 'diff'][category] = (costpost[year, DK, 'EVEH', 'post'][category]-costpost[year, DK, 'EVEH', 'pre'][category])/costpost[year, DK, 'EVEH', 'pre'][category]
            costpost[year, DK, 'EV', 'diff'][category] = (costpost[year, DK, 'EV', 'post'][category]-costpost[year, DK, 'EV', 'pre'][category])/costpost[year, DK, 'EV', 'pre'][category]
            costpost[year, DK, 'EH', 'diff'][category] = (costpost[year, DK, 'EH', 'post'][category]-costpost[year, DK, 'EH', 'pre'][category])/costpost[year, DK, 'EH', 'pre'][category]


costposttot=pd.DataFrame(columns=pd.MultiIndex.from_product([years,['EVEH','EV','EH']]),index=categories)
for year in years:
        for category in categories:
            costposttot[year, 'EVEH'][category]=(costpost[year, 'DK1', 'EVEH', 'diff'][category]+costpost[year, 'DK2', 'EVEH', 'diff'][category])/2
            costposttot[year, 'EV'][category]=(costpost[year, 'DK1', 'EV', 'diff'][category]+costpost[year, 'DK2', 'EV', 'diff'][category])/2
            costposttot[year, 'EH'][category]=(costpost[year, 'DK1', 'EH', 'diff'][category]+costpost[year, 'DK2', 'EH', 'diff'][category])/2
costposttot.to_csv('categsavings.csv')



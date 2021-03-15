import gdxpds as gd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



### This file reads optimal load scheduling from Julia ouput and generates aggregated
### consumption of alll categories, as in as in section 7.2.3 - step 2, following project report.
### Result is saved in csv and then converted to inc from xls2gdx ibrary in excel VBA.


# Sceanario and iteration
scenario='Expected'      ### choose between Expected, RapidScenario and SlowScenario
adoption=0.25            ### choose
iter='5it'               ### choose #it


# EV (categ,RRR,YYY,EH)
evnumbers=pd.read_csv('JuliatoBALMOREL/evsharesfin_'+scenario+'.csv',header=[0,1,2],index_col=0)
evnumbers.iloc[0:12]=evnumbers.iloc[0:12]*0.8 #houses only 80% at home
evnumbers.iloc[12:24]=evnumbers.iloc[12:24]*0.1 #apartments only 10% at home
categories=evnumbers.index
coefficients=pd.read_csv('JuliatoBALMOREL/evcoefficients.csv',skiprows=1,names=categories)/100 #[8*30]
years=['2020','2025','2030','2040']
lista=list(evnumbers.index.array)
lista.append('TOT')
EV=pd.DataFrame(index=range(8736),columns=range(1000))
EV.columns=pd.MultiIndex.from_product([lista,['DK1','DK2'],years,['BEV','BEVTOT','TOT','PHEV','PHEVTOT']])
TOTTOT=pd.DataFrame()
for DK in ['DK1','DK2']:
    for year in years:
        chargingBEV = pd.read_csv('JuliatoBALMOREL/Juliaout/FlexEV/FlexEV_' + DK + '_y' + year+ '.csv', header=None)
        chargingPHEV = pd.read_csv('JuliatoBALMOREL/Juliaout/FlexEV/FlexPHEV_' + DK + '_y' + year + '.csv', header=None)
        for category in evnumbers.index:
            weights=[evnumbers.loc[category][(DK,'BEV',year)],evnumbers.loc[category][(DK,'PHEV',year)]]
            EV[category,DK,year,'BEV']=sum(chargingBEV.iloc[:, i] * coefficients[category][i] for i in np.arange(20))
            EV[category,DK,year,'PHEV']=sum(chargingPHEV.iloc[:, i] * coefficients[category][i+20] for i in np.arange(10))
            EV[category, DK, year, 'TOT']=(EV[category,DK,year,'BEV'] * weights[0] +EV[category,DK,year,'PHEV'] * weights[1])/sum(weights)
            EV[category, DK, year, 'BEVTOT']=EV[category,DK,year,'BEV'] * evnumbers.loc[category][(DK,'BEV',year)]
            EV[category, DK, year, 'PHEVTOT']=EV[category,DK,year,'PHEV'] * evnumbers.loc[category][(DK,'PHEV',year)]
        EV['TOT', DK, year, 'BEVTOTT'] = EV[EV.columns[(EV.columns.get_level_values(1)==DK)&(EV.columns.get_level_values(2)==year) & (EV.columns.get_level_values(3)=='BEVTOT')]].sum(axis=1)
        EV['TOT', DK, year, 'PHEVTOTT'] = EV[EV.columns[(EV.columns.get_level_values(1)==DK)&(EV.columns.get_level_values(2)==year) & (EV.columns.get_level_values(3)=='PHEVTOT')]].sum(axis=1)
        TOTTOT[DK,year,'EV']=EV['TOT', DK, year, 'BEVTOTT']+EV['TOT', DK, year, 'PHEVTOTT']
# heavy
# EV.to_csv('JuliatoBALMOREL/EV'+scenario+str(adoption)+iter+'.csv')


# HP (categ,RRR,YYY,EH)
ehnumbers=pd.read_csv('JuliatoBALMOREL/ShareHP_categoriesfin_'+scenario+'.csv',header=[0,1,2],index_col=0)
EH=pd.DataFrame(index=range(8736),columns=range(800))
EH.columns=pd.MultiIndex.from_product([lista,['DK1','DK2'],years,['EH','AWTOT','AATOT','AASHTOT']]) # ('AA','AW','AASH')
for DK in ['DK1','DK2']:
    for year in years:
        AWprofile = pd.read_csv('JuliatoBALMOREL/Juliaout/FlexEH/FlexEHAW_' + DK + '_y' + year + '.csv', names=categories)
        AAprofile = pd.read_csv('JuliatoBALMOREL/Juliaout/FlexEH/FlexEHAA_' + DK + '_y' + year + '.csv', names=categories)
        AASHprofile = pd.read_csv('JuliatoBALMOREL/Juliaout/FlexEH/FlexEHSumHouse_' + DK + '_y' + year + '.csv',names=categories)
        for category in categories:
            weights=[ehnumbers.loc[category][DK, 'AW', year],ehnumbers.loc[category][DK, 'AA', year],ehnumbers.loc[category][DK, 'AASH', year]]
            #EH[category, DK, year, 'AA'] = (AWprofile[category] * weights[0] + AAprofile[category] * weights[1] +AASHprofile[category] * weights[2]) / sum(weights)  # change with share per each category
            #EH[category, DK, year, 'AW'] = (AWprofile[category] * weights[0] + AAprofile[category] * weights[1] +AASHprofile[category] * weights[2]) / sum(weights)  # change with share per each category
            #EH[category, DK, year, 'AASH'] = (AWprofile[category] * weights[0] + AAprofile[category] * weights[1] +AASHprofile[category] * weights[2]) / sum(weights)  # change with share per each category
            EH[category, DK, year, 'EH'] = (AWprofile[category] * weights[0] + AAprofile[category] * weights[1] +AASHprofile[category] * weights[2]) / sum(weights)  # change with share per each category
            EH[category, DK, year, 'AWTOT'] = AWprofile[category] * ehnumbers.loc[category][DK, 'AW', year]
            EH[category, DK, year, 'AATOT'] = AAprofile[category] * ehnumbers.loc[category][(DK, 'AA', year)]
            EH[category, DK, year, 'AASHTOT'] = AASHprofile[category] * ehnumbers.loc[category][(DK, 'AASH', year)]
        EH['TOT', DK, year, 'AWTOTT'] = EH[EH.columns[(EH.columns.get_level_values(1)==DK)&(EH.columns.get_level_values(2)==year) & (EH.columns.get_level_values(3)=='AWTOT')]].sum(axis=1)
        EH['TOT', DK, year, 'AATOTT'] = EH[EH.columns[(EH.columns.get_level_values(1)==DK)&(EH.columns.get_level_values(2)==year) & (EH.columns.get_level_values(3)=='AATOT')]].sum(axis=1)
        EH['TOT', DK, year, 'AASHTOTT'] = EH[EH.columns[(EH.columns.get_level_values(1)==DK)&(EH.columns.get_level_values(2)==year) & (EH.columns.get_level_values(3)=='AASHTOT')]].sum(axis=1)
        TOTTOT[DK,year,'EH']=EH['TOT', DK, year, 'AWTOTT']+EH['TOT', DK, year, 'AATOTT']+EH['TOT', DK, year, 'AASHTOTT']
# heavy
# EH.to_csv('JuliatoBALMOREL/EH'+scenario+str(adoption)+iter+'.csv')


# Yearly normalization
RESEVHP=pd.DataFrame()
ssn=[0,1680,3360,5040,8736]

RESEVHPDK1=pd.DataFrame()
RESEVHPDK2=pd.DataFrame()
for i in years:
    RESEVHPDK1[i]=pd.concat([TOTTOT['DK1',i,'EV'][ssn[p]:ssn[p+1]]-TOTTOT['DK1',i,'EV'][ssn[p]:ssn[p+1]].mean() for p in range(4)])+\
                    pd.concat([TOTTOT['DK1',i,'EH'][ssn[p]:ssn[p+1]]-TOTTOT['DK1',i,'EH'][ssn[p]:ssn[p+1]].mean() for p in range(4)])
    RESEVHPDK2[i]=pd.concat([TOTTOT['DK2',i,'EV'][ssn[p]:ssn[p+1]]-TOTTOT['DK2',i,'EV'][ssn[p]:ssn[p+1]].mean() for p in range(4)])+\
                    pd.concat([TOTTOT['DK2',i,'EH'][ssn[p]:ssn[p+1]]-TOTTOT['DK2',i,'EH'][ssn[p]:ssn[p+1]].mean() for p in range(4)])

RESEVHP['DK1']=pd.concat([RESEVHPDK1[i] for i in years])
RESEVHP['DK2']=pd.concat([RESEVHPDK2[i] for i in years])


# Adoption and to MW
# adoption= 0.40                 ### choose
RESEVHPOK=RESEVHP*adoption/1000

if scenario=='RapidScenario':
    RESEVHPOK['DK1']=RESEVHPOK['DK1']+TSRapDK1
    RESEVHPOK['DK2']=RESEVHPOK['DK2']+TSRapDK2
if scenario=='SlowScenario':
    RESEVHPOK['DK1']=RESEVHPOK['DK1']+TSSlowDK1
    RESEVHPOK['DK2']=RESEVHPOK['DK2']+TSSlowDK2

# Export
RESEVHPOK.to_csv('BALMOREL/RESEVHP files/xlsx/DE_RESEVHP'+scenario+str(adoption)+iter+'.csv') # normalized already but non adopted
RESEVHPOK.describe()


import gdxpds as gd
import pandas as pd
import numpy as np


### This file reads prices from Balmorel iteration and generates prices series
### for private economic optimization in Julia, as in section 7.2.3 - step 4,
### following project report

# Sceanario and iteration
scenario='Expected'      ### choose between Expected, RapidScenario and SlowScenario
adoption=0.25            ### choose
iter='5it'               ### choose #it



resultpath='BALMOREL\RESULTS run\MainResults'+scenario+adoption+iter[1]+'.gdx'
gams_dir='C:/GAMS/win64/32/'

# Reading prices and writing csv
varname = 'EL_PRICE_YCRST'
gamsprice = gd.to_dataframe(resultpath, varname, gams_dir=gams_dir, old_interface=False)
priceDK = gamsprice.loc[gamsprice.C == 'DENMARK']
for DK in ['DK1','DK2']:
  prices=pd.DataFrame()
  for year in [2020,2025,2030,2040]:
     serie=pd.Series()
     for week in priceDK['SSS'].unique():
         season=pd.concat([priceDK['Value'].loc[(priceDK.SSS==week) & (priceDK.RRR == DK) & (priceDK.Y == str(year))]], axis=0)
         serie=pd.concat([serie,season],ignore_index=True)
     prices[str(year)]=serie
  prices.to_csv('Julia/prices/price'+DK+scenario+adoption+iteration+'.csv',index=False)



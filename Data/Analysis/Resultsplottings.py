import gdxpds as gd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

gams_dir = 'C:/GAMS/win64/32/'
resultpath = 'C:/Users/pietr/Documents/GitHub/Tesi/BALMOREL/Slow_Adoption25%/MainResults_LS_1iter.gdx'
basepath='C:/Users/pietr/Documents/GitHub/Tesi/BALMOREL/Base/MainResults_LS_base.gdx'

# Define scenario and iteration
scenario = 'Expected25%'
iteration = '4iter'
runname = scenario


############################## RESULTS PLOTS ##############################

def extract(varname, runname=runname,
            DK=True):  # extracts from base and run into one for comparison, set DK=False for all countries
    base = gd.to_dataframe(basepath, varname, gams_dir=gams_dir, old_interface=False)
    run = gd.to_dataframe(resultpath, varname, gams_dir=gams_dir, old_interface=False)
    base['Model'] = 'Base'
    run['Model'] = runname
    tot = pd.concat([base, run])
    totDK = tot.loc[tot.C == 'DENMARK']
    if DK == False:
        return (tot)
    else:
        return (totDK)


# PLOT COSTS
datacosts = extract('OBJ_YCR')
costs = datacosts.SUBCATEGORY.unique()
plt.figure(figsize=(20, 10))
for type in list(enumerate(costs)):
    plt.subplot(3, 2, type[0] + 1)
    cost = datacosts.loc[datacosts.SUBCATEGORY == type[1]]
    sb.lineplot(cost.Y, cost.Value, hue=cost.RRR, style=cost.Model)
    plt.title(type[1])
    plt.xlabel('')
    plt.ylabel('M€')

# PLOT PRICES
priceDK = extract('EL_PRICE_YCRST')
years = priceDK.Y.unique()
plt.figure(figsize=(20, 10))
days = priceDK.shape[0] / 24
priceDK['Hour'] = pd.concat([pd.Series(np.arange(24))] * 448, axis=0).values
plt.figure()
for type in list(enumerate(years)):
    plt.subplot(2, 2, type[0] + 1)
    price = priceDK.loc[priceDK.Y == type[1]]
    sb.lineplot(price.Hour, price.Value, hue=price.Model, style=price.RRR, ci=False)
    plt.title(type[1])
    plt.xlabel('')
    plt.ylabel('€/MWh')


# PLOT EMISSIONS
Emissions = extract('EMI_YCRAG')
plt.figure(figsize=(8,8))
sb.barplot('Y','Value',hue='Model',data=Emissions, capsize=.2)
#sb.catplot('Y','Value',hue='Model',col='RRR',kind='bar',data=Emissions)
plt.title('Annual CO2-EQUIVALENT emissions')
plt.ylabel('ktons')